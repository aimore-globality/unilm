import json
from logging import Logger
from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import torch
import wandb
from pprint import pprint
from tqdm import tqdm, trange
import collections
from markuplmft.fine_tuning.run_swde import constants

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
from markuplmft.fine_tuning.run_swde.utils import set_seed, get_device_and_gpu_count
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Trainer:
    def __init__(
        self,
        model,
        no_cuda: bool,
        train_dataset_info,
        develop_dataset_info,
        per_gpu_train_batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        max_steps: int,
        logging_every_epoch: int,
        weight_decay,
        learning_rate,
        adam_epsilon,
        warmup_ratio,
        verbose: bool,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        output_dir: str,
        fp16: bool = True,
        fp16_opt_level: str = "O1",
        overwrite_output_dir: bool = False,
        evaluate_during_training: bool = False,
        save_every_epoch: int = 1,
    ):
        self.local_rank = -1
        self.fp16 = fp16
        self.device, self.n_gpu = get_device_and_gpu_count(no_cuda, self.local_rank)
        set_seed(self.n_gpu)  # ? For reproducibility

        # ? Setting Data
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = eval_batch_size

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.train_dataset, self.train_info = train_dataset_info
        self.develop_dataset, self.develop_info = develop_dataset_info

        #! Maybe pass this to a function before training prepare_for_training >
        self.training_dataloader_train = self._get_dataloader(self.train_dataset, True)

        self.max_steps = max_steps
        print(f"self.max_steps: {self.max_steps}")
        self.num_train_epochs = num_epochs
        if self.max_steps > 0:
            self.t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps
                // (len(self.training_dataloader_train) // self.gradient_accumulation_steps)
                + 1
            )
        else:
            self.t_total = (
                len(self.training_dataloader_train)
                // self.gradient_accumulation_steps
                * self.num_train_epochs
            )

        # ? Setting Model
        self.model = model
        self.model.to(self.device)
        self.optimizer = self._prepare_optimizer(
            self.model,
            weight_decay,
            learning_rate,
            adam_epsilon,
        )

        self.warmup_ratio = warmup_ratio
        self.scheduler = self._prepare_scheduler(self.optimizer, self.warmup_ratio, self.t_total)

        # ? Set model to use fp16 and multi-gpu
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                models=self.model, optimizers=self.optimizer, opt_level=fp16_opt_level
            )

        # ? multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # ? Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        self.max_grad_norm = max_grad_norm

        # self.logging_epoch = 0
        self.evaluate_during_training = evaluate_during_training
        self.logging_every_epoch = logging_every_epoch  #! Maybe change this to logging_epoch
        self.save_every_epoch = save_every_epoch

        self.verbose = verbose

        # ? Check if the folder exists
        self.output_dir = output_dir
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and not overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        #! < Maybe pass this to a function before training prepare_for_training
        if self.verbose:
            print("self.__dict__", pprint(self.__dict__))

    def _prepare_optimizer(self, model, weight_decay, learning_rate, adam_epsilon):
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

    def _prepare_scheduler(self, optimizer, warmup_ratio, t_total):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(warmup_ratio * t_total), num_training_steps=t_total
        )

    def _get_dataloader(self, dataset_info: pd.DataFrame, is_train: bool = True) -> DataLoader:

        if is_train:
            sampler = RandomSampler(dataset_info)
            batch_size = self.train_batch_size
        else:
            sampler = SequentialSampler(dataset_info)
            batch_size = self.eval_batch_size
        if self.local_rank != -1:
            sampler = DistributedSampler(dataset_info)

        loader = DataLoader(
            dataset=dataset_info,
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

    def _batch_to_device(self, batch_list):
        batch_list = tuple(batch.to(self.device) for batch in batch_list)
        return batch_list

    def train_epoch(self):
        self.model.train()
        all_losses = []
        epoch_iterator = tqdm(
            self.training_dataloader_train, desc="Batch", disable=self.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            batch_list = self._batch_to_device(batch)
            inputs = {
                "input_ids": batch_list[0],
                "attention_mask": batch_list[1],
                "token_type_ids": batch_list[2],
                "xpath_tags_seq": batch_list[3],
                "xpath_subs_seq": batch_list[4],
                "labels": batch_list[5],
            }
            outputs = self.model(**inputs)
            loss = outputs[0]  # ? model outputs are always tuple in transformers (see doc)

            if self.n_gpu > 1:
                loss = loss.mean()  # ? to average on multi-gpu parallel (not distributed) training
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.fp16:  # TODO (Aimore): Replace this with Accelerate
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # self.tr_loss += loss.item()  # ? Sum up new loss value
            all_losses.append(loss.item())

            # ? Optimize (update weights)
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

                # !Uncomment this
                # if (
                #     self.local_rank in [-1, 0]
                #     and self.logging_every_epoch > 0
                #     and self.global_step % self.logging_every_epoch == 0
                # ):  # ? Log metrics only once, not in the first step and at every global step
                #     if self.local_rank == -1 and self.evaluate_during_training:
                #         raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")

            # !Uncomment this
            # if 0 < self.max_steps < self.global_step:
            #     epoch_iterator.close()
            #     break

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
        print(f"- Training Loss: {np.mean(all_losses)}")

    def evaluate(self, dataset, info) -> Dict[str, float]:
        self.eval_dataloader = self._get_dataloader(dataset, False)
        if self.verbose:
            print(f"***** Running evaluation *****")
            # print(f"  Num examples for {website} = {len(data_loader.dataset)}")
            print(f"  Num examples for dataset = {len(self.eval_dataloader.dataset)}")
            print(f"  Batch size = {self.eval_batch_size}")
        self.model.eval()

        with torch.no_grad():
            all_logits = []
            all_losses = []
            tr_loss, logging_loss = 0.0, 0.0
            # epoch_iterator = tqdm(
            #     , desc="Batch", disable=self.local_rank not in [-1, 0]
            # )
            for batch in self.eval_dataloader:
                batch_list = self._batch_to_device(batch)
                inputs = {
                    "input_ids": batch_list[0],
                    "attention_mask": batch_list[1],
                    "token_type_ids": batch_list[2],
                    "xpath_tags_seq": batch_list[3],
                    "xpath_subs_seq": batch_list[4],
                    "labels": batch_list[5],  # ? Removing this won't report loss
                }
                outputs = self.model(**inputs)
                loss = outputs["loss"]  # model outputs are always tuple in transformers (see doc)
                logits = outputs["logits"]  # which is (bs,seq_len,node_type)
                all_logits.append(logits.detach().cpu())

                if self.n_gpu > 1:
                    # ? to average on multi-gpu parallel (not distributed) training
                    loss = loss.mean()

                # ! Not sure if I need this here to compute correctly the loss
                # if self.gradient_accumulation_steps > 1:
                #     loss = loss / self.gradient_accumulation_steps
                all_losses.append(loss.item())

        print(f"- Evaluation Loss: {np.mean(all_losses)}")

        return self.recreate_dataset(all_logits, info)

    def recreate_dataset(self, all_logits, info):
        all_probs = torch.softmax(
            torch.cat(all_logits, dim=0), dim=2
        )  # (all_samples, seq_len, node_type)

        assert len(all_probs) == len(info)

        all_res = collections.defaultdict(dict)

        for sub_prob, sub_info in zip(all_probs, info):
            (
                html_path,
                involved_first_tokens_pos,
                involved_first_tokens_xpaths,
                involved_first_tokens_types,
                involved_first_tokens_text,
                involved_first_tokens_gt_text,
            ) = sub_info

            for pos, xpath, type, text, gt_text in zip(
                involved_first_tokens_pos,
                involved_first_tokens_xpaths,
                involved_first_tokens_types,
                involved_first_tokens_text,
                involved_first_tokens_gt_text,

            ):

                pred = sub_prob[pos]  # ? This gets the first logit of each respective node
                # ? sub_prob = [tensor([0.0045, 0.9955]), ...], sub_prob.shape = [384, 2]
                # ? pos = 14
                # ? pred = tensor([0.0045, 0.9955])
                if xpath not in all_res[html_path]:
                    all_res[html_path][xpath] = {}
                    all_res[html_path][xpath]["pred"] = pred
                    all_res[html_path][xpath]["truth"] = type
                    all_res[html_path][xpath]["text"] = text
                    all_res[html_path][xpath]["gt_text"] = gt_text

                else:
                    all_res[html_path][xpath]["pred"] += pred
                    assert all_res[html_path][xpath]["truth"] == type
                    assert all_res[html_path][xpath]["text"] == text

        lines = {
            "html_path": [],
            "xpath": [],
            "text": [],
            "gt_text": [],
            "truth": [],
            "pred_type": [],
            "final_probs": [],
        }

        for html_path in all_res:
            # E.g. all_res [dict] = {html_path = {xpath = {'pred': tensor([0.4181, 0.5819]), 'truth': 'PAST_CLIENT', 'text': 'A healthcare client gains control of their ACA processes | BerryDunn'},...}, ...}
            for xpath in all_res[html_path]:
                final_probs = all_res[html_path][xpath]["pred"] / torch.sum(
                    all_res[html_path][xpath]["pred"]
                )  # TODO(aimore): Why is this even here? torch.sum(both prob) will always be 1, what is the point then? Maybe in case of more than one label?
                pred_id = torch.argmax(final_probs).item()
                pred_type = constants.ATTRIBUTES_PLUS_NONE[pred_id]
                final_probs = final_probs.numpy().tolist()

                lines["html_path"].append(html_path)
                lines["xpath"].append(xpath)
                lines["gt_text"].append(all_res[html_path][xpath]["gt_text"])
                lines["text"].append(all_res[html_path][xpath]["text"])
                lines["truth"].append(all_res[html_path][xpath]["truth"])
                lines["pred_type"].append(pred_type)
                lines["final_probs"].append(final_probs)

        result_df = pd.DataFrame(lines)
        return result_df

    def train(self):

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
            print(f"Epoch: {epoch}")
            self.evaluate(self.develop_dataset, self.develop_info)

            if isinstance(self.training_dataloader_train, DataLoader) and isinstance(
                self.training_dataloader_train.sampler, DistributedSampler
            ):
                self.training_dataloader_train.sampler.set_epoch(epoch)

            self.train_epoch()
            # print(f"Epoch: {epoch} - Train Loss: {self.tr_loss / self.global_step}")

            #! Add evaluation here
            # if self.evaluate_during_training:
            #     self.evaluate(self.develop_dataset, self.develop_info)
            # self._log_metrics(develop_metrics, prefix="develop")

            if self.global_step > self.max_steps:  #! I have changed this
                train_iterator.close()
                break

        #!Add  # Save the trained model and the tokenizer
        # if (self.local_rank == -1 or torch.distributed.get_rank() == 0):
        #     self.save_model_and_tokenizer()

        print("Done training")
        self.model.eval()

        dataset_evaluated = self.evaluate(self.develop_dataset, self.develop_info)
        return dataset_evaluated
