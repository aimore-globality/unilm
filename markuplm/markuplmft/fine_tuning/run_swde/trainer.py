import json
from logging import Logger
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from pprint import pprint
from tqdm import tqdm, trange

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )
import random

# from microcosm_logging.decorators import logger
# from microcosm_sagemaker.metrics.experiment_metrics import ExperimentMetrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.functional import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def set_seed(n_gpu):
    """
    Fix the random seed for reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

seed = 66 #! Set the seed like other modules

class Trainer:


    def __init__(
        self,
        model,
        no_cuda:bool,
        train_dataset,
        develop_dataset,
        per_gpu_train_batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        weight_decay,
        learning_rate,
        adam_epsilon,
        warmup_ratio,
        verbose: bool,
        gradient_accumulation_steps: int,
        output_dir:str,
        fp16:bool=True,
        fp16_opt_level: str="O1",
    ):
        self.local_rank = -1
        self.fp16 = fp16
        self._setup_cuda_gpu(no_cuda)
        set_seed(self.n_gpu)

        self.model = model
        self.num_train_epochs = num_epochs

        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = eval_batch_size

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = self.prepare_optimizer(self.model, weight_decay, learning_rate, adam_epsilon)

        if self.max_steps > 0:
            self.t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps // (len(self.train_dataloader) // self.gradient_accumulation_steps) + 1
            )
        else:
            self.t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        self.warmup_ratio = warmup_ratio
        self.scheduler = self.prepare_scheduler(self.optimizer, self.warmup_ratio, self.t_total)


        #? Set model to use fp16 and multi-gpu
        if self.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)

        #? multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        #? Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        # self.logging_epoch = 0
        self.save_steps = 1000

        self.verbose = verbose

        # ? Check if the folder exists
        self.output_dir = output_dir
        if (
                os.path.exists(self.output_dir)
                and os.listdir(self.output_dir)
                and not self.overwrite_output_dir
            ):
                raise ValueError(f"Output directory ({self.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

        if self.verbose:
            print("self.__dict__", pprint(self.__dict__))

        self.train_dl = self._get_dataloader(train_dataset, True)
        self.develop_dl = self._get_dataloader(develop_dataset, False)

    def _setup_cuda_gpu(self, no_cuda):
        #? Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count() if not no_cuda else 0
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

    def prepare_optimizer(self, model, weight_decay, learning_rate, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for param_name, param in model.named_parameters()
                    if not any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for param_name, param in model.named_parameters()
                    if any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    def prepare_scheduler(optimizer, warmup_ratio, t_total):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(warmup_ratio * t_total), num_training_steps=t_total
        )

    def _get_dataloader(self, dataset: pd.DataFrame, is_train: bool = True) -> DataLoader:

        if is_train:
            sampler = RandomSampler(dataset)
            batch_size = self.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = self.eval_batch_size
        if self.local_rank != -1:
            sampler = DistributedSampler(dataset)

        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
        )
        return loader

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        print(f"Logs for epoch={self.logging_epoch}")
        if prefix:
            metrics = {
                f"{prefix}_{metric}": value
                for metric, value in metrics.items()
            }
        print(json.dumps(metrics, indent=4))
        if self.metrics_store:
            self.metrics_store.log_timeseries(**metrics, step=self.logging_epoch)

    def _to_device(self, tensors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            key: tensor.to(self.device)
            for key, tensor in tensors.items()
        }

    def batch_to_device(self, batch_list):
        batch_list = tuple(batch.to(self.device) for batch in batch_list)
        return batch_list

    def train_epoch(self):
        print(f"Starting epoch {self.logging_epoch}")
        self.model.train()
        batch_losses = []
        score_batches = []
        target_batches = []

        epoch_iterator = tqdm(
            self.train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(
            epoch_iterator
        ):  # tqdm(self.train_dl, total=len(self.train_dl))

            batch_list = self.batch_to_device(batch)
            inputs = {
                "input_ids": batch_list[0],
                "attention_mask": batch_list[1],
                "token_type_ids": batch_list[2],
                "xpath_tags_seq": batch_list[3],
                "xpath_subs_seq": batch_list[4],
                "labels": batch_list[5],
            }
            outputs = self.model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if self.n_gpu > 1:
                loss = loss.mean()  # to average on multi-gpu parallel (not distributed) training
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.fp16:  # TODO (Aimore): Replace this with Accelerate
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), self.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.global_step += 1

                if (
                    self.local_rank in [-1, 0]
                    and self.logging_steps > 0
                    and self.global_step % self.logging_steps == 0
                ):
                    # Log metrics
                    if self.local_rank == -1 and self.evaluate_during_training:
                        raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")

            if 0 < self.max_steps < self.global_step:
                epoch_iterator.close()
                break


            self.logger.debug(f"Batch loss: {loss}")
            batch_losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()


            score_batches.append(similarities_normalized.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())

        all_scores = np.concatenate(score_batches)
        all_targets = np.concatenate(target_batches)
        metrics = None

        # if wandb.run:
            # wandb.log(
            #     {"scores": wandb.Histogram(all_scores)},
            #     step=self.logging_epoch,
            # )
        # self._log_metrics(
        #     {
        #         "loss": np.mean(batch_losses),
        #         "current_lr": self.optimizer.param_groups[0]["lr"],
        #         **metrics,
        #     },
        #     prefix="train",
        # )

    def calculate_metrics(self, scores: np.array, targets: np.array) -> Dict[str, float]:
        pass

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        proba_batches = []
        target_batches = []
        losses = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = self._to_device(batch)
                targets = batch.pop("labels")
                similarities_normalized = self.model(batch)
                probas = self.model.head.calculate_predictions(similarities_normalized)
                proba_batches.append(probas.cpu().numpy())
                target_batches.append(targets.cpu().numpy())
                loss = self.model.head.calculate_loss(similarities_normalized, targets)
                losses.append(loss.item())
        all_probas = np.concatenate(proba_batches)
        all_targets = np.concatenate(target_batches)
        metrics = None
        metrics["loss"] = np.mean(losses)
        return metrics

    def train(self):
        # develop_metrics = self.evaluate(self.develop_dl)
        # self._log_metrics(develop_metrics, "develop")

        print("***** Running training *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Num Epochs = {self.num_train_epochs}")
        print(f"  Instantaneous batch size per GPU = {self.per_gpu_train_batch_size}")
        total_train_batch_size = (
            self.train_batch_size
            * self.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.local_rank != -1 else 1)
        )
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        print(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.t_total}")

        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0

        train_iterator = trange(
            int(self.num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0]
        )
        for epoch in train_iterator:  # range(1, self.num_epochs + 1):
            if isinstance(self.train_dataloader, DataLoader) and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.logging_epoch = epoch
            self.train_epoch()

            if self.global_step > self.max_steps: #! I have changed this
                train_iterator.close()
                break
            # develop_metrics = self.evaluate(self.develop_dl)
            # self._log_metrics(develop_metrics, prefix="develop")

        #!Add  # Save the trained model and the tokenizer
        if self.do_train and (self.local_rank == -1 or torch.distributed.get_rank() == 0):
            self.save_model_and_tokenizer()

        print("Done training")
        self.model.eval()
        # store pre-computed label embeddings for fast inference
        self.model.label_embeddings = self.model.encode_labels()
