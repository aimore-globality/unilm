# import json
# from logging import Logger
# from typing import Dict, List, Optional

# import numpy as np
# import pandas as pd
# import torch
# import wandb
# from microcosm_logging.decorators import logger
# from microcosm_sagemaker.metrics.experiment_metrics import ExperimentMetrics
# from sklearn.preprocessing import MultiLabelBinarizer
# from torch import nn
# from torch.functional import Tensor
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import get_linear_schedule_with_warmup

# from service_offering_classifier.bundles.common.metrics import MetricsCalculator
# from service_offering_classifier.bundles.dbc.dataset import ServiceOfferingClassificationDataset
# from service_offering_classifier.bundles.dbc.model import DescriptionBasedClassifier
from __future__ import absolute_import, division, print_function
import multiprocessing as mp
import argparse
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset

# import copy
import glob
import logging
import os
import random
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
# from markuplmft.fine_tuning.run_swde.eval_utils import page_level_constraint
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from markuplmft.fine_tuning.run_swde.utils import SwdeDataset, get_swde_features

from markuplmft.fine_tuning.run_swde import constants
from markuplmft.models.markuplm import (
    MarkupLMConfig,
    MarkupLMForTokenClassification,
    MarkupLMTokenizer,

)
from sklearn.metrics import confusion_matrix

try:
    from apex import amp                
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )

# @logger
class Trainer:
    def __init__(
        self,
        model: DescriptionBasedClassifier,
        device: torch.device,
        train_data: pd.DataFrame,
        develop_data: pd.DataFrame,
        train_batch_size: int,
        eval_batch_size: int,
        lr: float,
        num_epochs: int,
        warmup_steps: int,
        max_grad_norm: float,
        head_lr_multiplier: float,
        metrics_top_k: int,
        metrics_store: Optional[ExperimentMetrics] = None,
    ):


        self.model = None
        self.tokenizer = None
        self.no_cuda = False
        self.local_rank = -1
        self.fp16 = True
        self.fp16_opt_level = "O1"
        self.weight_decay = 0.0
        self.warmup_ratio = 0.0
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8

        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-5
        self.adam_epsilon = 1e-8
        self.num_train_epochs = 1
        self.max_grad_norm = 1.0

        self.save_steps = 3000
        self.max_step = -1
        self.pre_trained_model_folder_path = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/286"
        self.evaluate_during_training = False
        self.logging_steps = 10
        self.output_dir = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/"
        self.overwrite_output_dir = True
        
        self.model_name_or_path = "microsoft/markuplm-base"
        self.save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/"

        self.data_root_dir = "/data/GIT/swde/my_data/train/my_CF_processed/"
        self.train_data_root_dir = "/data/GIT/swde/my_data/train/my_CF_processed/"
        self.develop_data_root_dir = "/data/GIT/swde/my_data/develop/my_CF_processed/"

        self.doc_stride = 128
        self.max_seq_length = 384
        self.overwrite_cache = False
        self.save_features = True

        self.do_train = True
        self.parallelize = True
        self.max_steps = -1






        # self.model = model
        # self.device = device

        # self.train_batch_size = train_batch_size
        # self.eval_batch_size = eval_batch_size
        # self.lr = lr
        # self.num_epochs = num_epochs
        # self.max_grad_norm = max_grad_norm
        # self.head_lr_multiplier = head_lr_multiplier
        # self.metrics_top_k = metrics_top_k

        # self.label_binarizer = MultiLabelBinarizer(classes=model.label_ids).fit(None)
        # self.train_dl = self._get_dataloader(train_data, True)
        # self.develop_dl = self._get_dataloader(develop_data, False)

        # train_label_frequencies = self.label_binarizer.transform(train_data.label).mean(axis=0)
        # self.model.head.initialize(train_label_frequencies)

        # # setting a different learning rate for the params of the model "head"
        # all_params: List[Dict] = [
        #     {"params": self.model.encoder.parameters()},
        #     {
        #         "params": self.model.head.parameters(),
        #         "lr": self.lr * self.head_lr_multiplier,
        #     },
        # ]

        # self.optimizer = torch.optim.AdamW(all_params, lr=self.lr)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_training_steps=num_epochs * len(self.train_dl),
        #     num_warmup_steps=warmup_steps,
        # )
        # self.logging_epoch = 0
        # self.logger: Logger  # for mypy's benefit
        # self.metrics_store = metrics_store

    def _get_dataloader(self, data: pd.DataFrame, is_train: bool = True) -> DataLoader:
        batch_size = self.train_batch_size if is_train else self.eval_batch_size
        texts = data.text.tolist()
        tensorized_inputs = self.model.tensorize_inputs(texts)
        encoded_labels = Tensor(self.label_binarizer.transform(data.label))
        dataset = ServiceOfferingClassificationDataset(tensorized_inputs, encoded_labels)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            pin_memory=True,
        )
        return loader

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        self.logger.info(f"Logs for epoch={self.logging_epoch}")
        if prefix:
            metrics = {
                f"{prefix}_{metric}": value
                for metric, value in metrics.items()
            }
        self.logger.info(json.dumps(metrics, indent=4))
        if self.metrics_store:
            self.metrics_store.log_timeseries(**metrics, step=self.logging_epoch)

    def _to_device(self, tensors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            key: tensor.to(self.device)
            for key, tensor in tensors.items()
        }

    def train_epoch(self):
        self.logger.info(f"Starting epoch {self.logging_epoch}")
        self.model.train()
        batch_losses = []
        score_batches = []
        target_batches = []

        for batch in tqdm(self.train_dl, total=len(self.train_dl)):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            targets = batch.pop("labels")
            similarities_normalized = self.model(batch)
            loss = self.model.head.calculate_loss(similarities_normalized, targets)
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
        metrics = self.calculate_metrics(all_scores, all_targets)
        if wandb.run:
            wandb.log(
                {"scores": wandb.Histogram(all_scores)},
                step=self.logging_epoch,
            )
        self._log_metrics(
            {
                "loss": np.mean(batch_losses),
                "current_lr": self.optimizer.param_groups[0]["lr"],
                **metrics,
            },
            prefix="train",
        )

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
        metrics = self.calculate_metrics(all_probas, all_targets)
        metrics["loss"] = np.mean(losses)
        return metrics

    def train(self):
        develop_metrics = self.evaluate(self.develop_dl)
        self._log_metrics(develop_metrics, "develop")
        self.logger.info("Starting to train")

        for epoch in range(1, self.num_epochs + 1):
            self.logging_epoch = epoch
            self.train_epoch()
            develop_metrics = self.evaluate(self.develop_dl)
            self._log_metrics(develop_metrics, prefix="develop")
        self.logger.info("Done training")
        self.model.eval()
        # store pre-computed label embeddings for fast inference
        self.model.label_embeddings = self.model.encode_labels()
