from typing import Dict
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
import wandb

from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset

from torch.utils.data import DataLoader
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.utils import set_seed
from transformers import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
from pathlib import Path
import multiprocess as mp

from accelerate import Accelerator
from transformers import AdamW, get_scheduler


class Trainer:
    def __init__(
        self,
        model,
        no_cuda: bool,
        train_dataset,
        evaluate_dataset,
        per_gpu_train_batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        max_steps: int,
        logging_every_epoch: int,
        weight_decay: float,
        learning_rate: float,
        adam_epsilon: float,
        warmup_ratio: float,
        verbose: bool,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        save_model_path: str,
        featurizer,
        overwrite_model: bool = False,
        fp16: bool = True,
        fp16_opt_level: str = "O1",
        evaluate_during_training: bool = False,
        save_every_epoch: int = 1,
        local_rank: int = -1,
        device=None,
        n_gpu: int = 0,
        run=None,
        just_evaluation=False,
    ):
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.just_evaluation = just_evaluation
        set_seed(self.n_gpu)  # ? For reproducibility
        self.featurizer = featurizer
        self.logging_epoch = 0

        # ? Setting WandB Log
        if run:
            self.run = run
        else:
            defaults = {}
            self.run = wandb.init(project="LanguageModel", config=defaults, resume=True)

        # ? Setting Data
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = eval_batch_size

        self.verbose = verbose

        self.train_dataset = train_dataset
        self.evaluate_dataset = evaluate_dataset

        self.train_dataset["html"] = self.train_dataset["html"].astype("category")
        self.evaluate_dataset["html"] = self.evaluate_dataset["html"].astype("category")

        self.train_dataloader = self._get_dataloader(self.featurizer.feature_to_dataset(self.train_dataset["swde_features"].explode().values), is_train=True)

        self.develop_dataloader = self._get_dataloader(self.featurizer.feature_to_dataset(self.evaluate_dataset["swde_features"].explode().values), is_train=False)
        #! Maybe pass this to a function before training prepare_for_training >

        self.max_steps = max_steps
        print(f"self.max_steps: {self.max_steps}")
        self.num_train_epochs = num_epochs

        # ? Setting Model
        self.model = model
        self.num_training_steps = num_epochs * len(self.train_dataloader)

        self.optimizer = AdamW(self.model.parameters(), lr=3e-5)
        
        training_samples = len(train_dataset)
        evaluation_samples = len(evaluate_dataset)
        if self.run:
            self.run.log(
                {"training_samples": training_samples, "evaluation_samples": evaluation_samples}
            )

        print(f"training_samples: {training_samples}")
        print(f"evaluation_samples: {evaluation_samples}")
        print("... Done!")

    def _get_dataloader(
        self, dataset_featurized: pd.DataFrame, is_train: bool = True
    ) -> DataLoader:
        if is_train:
            shuffle=True
            batch_size = self.train_batch_size
        else:
            shuffle=False
            batch_size = self.eval_batch_size

        loader = DataLoader(
            dataset=dataset_featurized,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=1
        )
        return loader      

    def train(self):
        print("***** Running training *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Num Epochs = {self.num_train_epochs}")
        print(f"  Instantaneous batch size per GPU = {self.per_gpu_train_batch_size}")

        self.tr_loss, self.logging_loss = 0.0, 0.0

        self.accelerator = Accelerator()
        device = self.accelerator.device
        self.model.to(device)

        (
            self.train_dataloader,
            self.develop_dataloader,
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.train_dataloader, self.develop_dataloader, self.model, self.optimizer
        )
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )
        progress_bar = tqdm(range(self.num_training_steps), disable=not self.accelerator.is_main_process)
        for epoch in progress_bar:
            self.model.train()
            all_losses = []
            for setp, batch in enumerate(self.train_dataloader):
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],  # Uncomment this to use Roberta
                    "labels": batch[3],
                }
            outputs = self.model(**inputs)
            loss = outputs.loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()
            progress_bar.update(1)
            all_losses.append(loss.item())

            # Evaluation step
            
        self.accelerator.print(f"Epoch: {epoch}")