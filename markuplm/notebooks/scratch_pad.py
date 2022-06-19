# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.11 ('markuplmft')
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import pandavro as pdx
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
from markuplmft.fine_tuning.run_swde.featurizer import SwdeDataset
from transformers import RobertaTokenizer
import torch


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

DOC_STRIDE = 128
MAX_SEQ_LENGTH = 384
featurizer = Featurizer(tokenizer=tokenizer, doc_stride=DOC_STRIDE, max_length=MAX_SEQ_LENGTH)

# %%
import glob

develop_domains_path = glob.glob(
        f"/data/GIT/delete-abs/develop/processed_dedup/*.pkl"
    )
print(f"develop_domains_path: {len(develop_domains_path)} - {develop_domains_path[0]}")

# %%
df_develop = pd.DataFrame()
for domain_path in develop_domains_path:
    df_develop = df_develop.append(pd.read_pickle(domain_path))

# %%
df_nodes = df_develop.explode("nodes").reset_index()
# # ? Join expanded nodes into df
df_nodes = df_nodes.join(
    pd.DataFrame(
        df_nodes.pop("nodes").tolist(),
        columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
    )
)

# %%
df_nodes["node_tok_ids"] = df_nodes["node_text"].apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))

# %%
df_nodes['node_tok_ids_len'] = df_nodes['node_tok_ids'].apply(len).values

# %%
df_nodes

# %%
big_nodes = df_nodes[df_nodes['node_tok_ids_len'] > 128][["node_tok_ids_len", "node_text", "node_gt_tag", "node_gt_text"]].sort_values(["node_tok_ids_len"], ascending=False)

# %%
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
print(encoded_input)
# {'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
#                [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
#                [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
#                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#                     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}


# %%

# %%
batch_sentences = [
    "But what about second breakfast?",
    "Don't think hehink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Piphink he knows about second breakfast, Pip knows about second breakfast, Pip.",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
tokenizer.prepare_for_model()
# print(encoded_input)
# {'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
#                [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
#                [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
#                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#                     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}


# %%
tokenizer2.prepare_for_model(tokenizer2.convert_tokens_to_ids(tokenizer.tokenize(df_nodes["node_text"].iloc[0])))


# %%
from transformers import AutoTokenizer
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.prepare_for_model(df_nodes["node_tok_ids"].iloc[0])['input_ids']
df_nodes["node_tok_ids"].iloc[0]

# %%
big_nodes[["node_tok_ids_len", "node_text", "node_gt_tag", "node_gt_text"]].sort_values(["node_tok_ids_len"], ascending=False)[big_nodes['node_gt_tag']=='PAST_CLIENT']

# %%
# t = 
for t in range(222):
    gt_text = big_nodes[["node_tok_ids_len", "node_text", "node_gt_tag", "node_gt_text"]].sort_values(["node_tok_ids_len"], ascending=False)[big_nodes['node_gt_tag']=='PAST_CLIENT'].iloc[t]["node_gt_text"]
    # print(gt_text)
    seg_text = big_nodes[["node_tok_ids_len", "node_text", "node_gt_tag"]].sort_values(["node_tok_ids_len"], ascending=False)[big_nodes['node_gt_tag']=='PAST_CLIENT'].iloc[t].values[1].split('. ')
    # print(seg_text)

    ok = []
    for x in gt_text:
        for y in seg_text:
            if x.lower() in y.lower():
                ok.append(x)
                break
        if len(y) > 128:
            break

    if len(gt_text) != len(ok):
        print("ok:", ok)
        print("gt_text:", gt_text)
        print("ERROR ")
        print(gt_text)
        print(seg_text)
        break

# %%
df_develop

# %%
df["page_features"] = df.apply(
            lambda page: featurizer.get_page_features(page["url"], page["nodes"]), axis=1
        )

# %%
# df = pd.read_pickle("/data/GIT/delete/develop/prepared/1820productions.com.pkl")
# df = pd.read_pickle("/data/GIT/delete/develop/1820productions.com.pkl")
df = pd.read_pickle("/data/GIT/delete/develop/4-most.co.uk.pkl")
print(f"Memory: {sum(df.memory_usage(deep=True))/10**6:.2f} Mb")
df.head(3)

# %%
# df.apply(lambda x: featurizer.get_swde_features(x['nodes'], x['url']),axis=1)

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import transformers
model  = transformers.RobertaForTokenClassification.from_pretrained('roberta-base')
# model  = transformers.RobertaForTokenClassification('roberta-base')

# %%
import wandb
import os
import torch
from markuplmft.fine_tuning.run_swde.utils import get_device_and_gpu_count

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank=-1

print(f"local_rank: {local_rank}")

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

no_cuda = False
local_rank=-1

device, n_gpu = get_device_and_gpu_count(no_cuda, local_rank)
print(device, n_gpu)

# %%
trainer_config = dict(
    # # ? Optimizer
    weight_decay= 0.01, #? Default: 0.0
    learning_rate=1e-05,  #? Default: 1e-05
    adam_epsilon=1e-8, #? Default: 1e-8
    # # ? Loss
    # label_smoothing=0.01, #? Default: 0.0 
    # loss_function = "CrossEntropyLoss", #? Default: CrossEntropyLoss / FocalLoss
    # # ? Scheduler
    warmup_ratio=0.0, #? Default: 0
    # # ? Trainer
    num_epochs = 1, 
    gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
    max_steps = 0, 
    per_gpu_train_batch_size = int(16), #? 34 Max with the big machine 
    eval_batch_size = int(512), #? 1024 Max with the big machine 
    fp16 = True, 
    fp16_opt_level = "O1",
    max_grad_norm = 1.0,
    # load_model=False,
    # load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_2/checkpoint-2",
    # freeze_body = False,
    save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
    overwrite_model = True,
    evaluate_during_training = True,
    no_cuda = no_cuda,
    verbose = False,
    logging_every_epoch = 1,
    # # ? Data Reader
    # dataset_to_use='all',
    # overwrite_cache=True, 
    # parallelize=False, 
    # train_dedup=True, #? Default: False
    # develop_dedup=True, #? Default: False
)

# %%
from typing import Dict
import numpy as np
import pandas as pd
import torch
import wandb
from pprint import pprint
from tqdm import tqdm
import collections
from markuplmft.fine_tuning.run_swde import constants
import wandb

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset

from torch.functional import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.utils import set_seed
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
from pathlib import Path
import multiprocess as mp

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
        self.fp16 = fp16
        self.device = device
        self.n_gpu = n_gpu
        self.just_evaluation = just_evaluation
        set_seed(self.n_gpu)  #? For reproducibility

        self.featurizer = featurizer
        self.logging_epoch = 0
        
        # #? Setting WandB Log
        if run:
            self.run = run
        else:
            if self.local_rank in [-1, 0]:
                defaults = {}                
                self.run = wandb.init(project="LanguageModel", config=defaults, resume=True)
            else:
                self.run = None

        # #? Setting Data
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = eval_batch_size

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.verbose = verbose

        self.train_dataset = train_dataset
        self.evaluate_dataset = evaluate_dataset

        
        self.train_dataloader, _ = self._get_dataloader(self.train_dataset, is_train=True)
        self.evaluate_dataloader, _ = self._get_dataloader(self.evaluate_dataset, is_train=False)
        # #! Maybe pass this to a function before training prepare_for_training >

        self.max_steps = max_steps
        print(f"self.max_steps: {self.max_steps}")
        self.num_train_epochs = num_epochs
        if self.max_steps > 0:
            self.t_total = self.max_steps
            self.num_train_epochs = int(
                (
                    self.max_steps
                    // (len(self.train_dataloader) // self.gradient_accumulation_steps)
                    + 1
                )
            )
        else:
            self.t_total = (
                len(self.train_dataloader)
                // self.gradient_accumulation_steps
                * self.num_train_epochs
            )

        # #? Setting Model
        self.model = model
        print(f"self.device:{self.device}")
        self.model.to(self.device)
        self.optimizer = self._prepare_optimizer(
            self.model,
            weight_decay,
            learning_rate,
            adam_epsilon,
        )

        self.warmup_ratio = warmup_ratio
        self.scheduler = self._prepare_scheduler(self.optimizer, self.warmup_ratio, self.t_total)

        # #? Set model to use fp16 and multi-gpu
        if self.fp16 and not self.no_cuda:
            self.model, self.optimizer = amp.initialize(
                models=self.model, optimizers=self.optimizer, opt_level=fp16_opt_level
            )

        # #? multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # #? Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        self.max_grad_norm = max_grad_norm

        self.evaluate_during_training = evaluate_during_training
        self.logging_every_epoch = logging_every_epoch  #! Maybe change this to logging_epoch
        self.save_every_epoch = save_every_epoch

        # #? Check if the folder exists
        self.overwrite_model = overwrite_model

        self.save_model_path = Path(save_model_path) / f"epochs_{str(self.num_train_epochs)}"
        if (
            self.save_model_path.exists()
            and any(self.save_model_path.iterdir())  #? Check if the folder is empty
            and not self.overwrite_model
        ):
            raise ValueError(
                f"Output directory ({self.save_model_path}) already exists and is not empty. Use --overwrite_model to overcome."
            )        

        training_samples = len(train_dataset)
        evaluation_samples = len(evaluate_dataset)
        if self.run:
            self.run.log(
                {"training_samples": training_samples, "evaluation_samples": evaluation_samples}
            )

        print(f"training_samples: {training_samples}")
        print(f"evaluation_samples: {evaluation_samples}")
        print(f"learning_rate: {learning_rate}")
        #! < Maybe pass this to a function before training prepare_for_training
        print("... Done!")

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

    def _get_dataloader(self, dataset: pd.DataFrame, is_train: bool = True) -> DataLoader:
        features = dataset["swde_features"].explode().values
        dataset_featurized = self.featurizer.feature_to_dataset(features)

        print(f"Get dataloader for dataset: {len(dataset_featurized)}")
        if is_train:
            sampler = RandomSampler(dataset_featurized)
            batch_size = self.train_batch_size
        else:
            sampler = SequentialSampler(dataset_featurized)
            batch_size = self.eval_batch_size
        if self.local_rank != -1:
            sampler = DistributedSampler(dataset_featurized)

        num_workers = int(mp.cpu_count() / torch.cuda.device_count())

        loader = DataLoader(
            dataset=dataset_featurized,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            # generator="forkserver", #? According to this: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#:~:text=If%20you%20plan,change%20this%20setting.
            # multiprocessing_context= "forkserver", #? According to this: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#:~:text=If%20you%20plan,change%20this%20setting.
        )
        return loader, dataset_featurized

    def _batch_to_device(self, batch_list):
        batch_list = tuple(batch.to(self.device) for batch in batch_list)
        return batch_list

    def train_epoch(self):
        self.model.train()
        all_losses = []

        for step, batch in enumerate(self.train_dataloader):
            # batch_iterator.update()
            # batch_iterator.set_description(f"Training on Batch:")
            # print(f"step = {step} | self.local_rank: {self.local_rank}")

            batch_list = self._batch_to_device(batch)
            inputs = {
                "input_ids": batch_list[0],
                "attention_mask": batch_list[1],
                "token_type_ids": batch_list[2],
                "labels": batch_list[3],
            }
            outputs = self.model(**inputs)
            loss = outputs[0]  #? model outputs are always tuple in transformers (see doc)

            if self.n_gpu > 1:
                loss = loss.mean()  #? to average on multi-gpu parallel (not distributed) training
                # loss = loss.sum() / len(batch) #! See this to understand that loss.mean() is not optimal: https://discuss.pytorch.org/t/how-to-fix-gathering-dim-0-warning-in-multi-gpu-dataparallel-setting/41733/2#:~:text=Actually%2C%20just%20re,sizes%20as%20well).
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.fp16 and not self.no_cuda:  # TODO (Aimore): Replace this with Accelerate
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # self.tr_loss += loss.item()  #? Sum up new loss value
            all_losses.append(loss.item())

            # #? Optimize (update weights)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16 and not self.no_cuda:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), self.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.global_step += 1

                # # # !Uncomment this
                # if (
                #     self.local_rank in [-1, 0]
                #     and self.logging_every_epoch > 0
                #     and self.global_step % self.logging_every_epoch == 0
                # ):  #? Log metrics only once, not in the first step and at every global step
                #     if self.local_rank == -1 and self.evaluate_during_training:
                #         raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")

            # # !Uncomment this
            # if 0 < self.max_steps < self.global_step:
            #     print(
            #         "Forced break because self.max_steps ({self.max_steps}) > self.global_step ({self.global_step})"
            #     )
            #     batch_iterator.close()
            #     break

        print(f"- Training Loss: {np.mean(all_losses)}")
        if self.local_rank in [-1, 0]:
            self.logging_epoch += 1
            self.run.log(
                {"lr": self.scheduler.get_last_lr()[0], "global_step": self.global_step},
                step=self.logging_epoch,
            )
            self.run.log({"Training_Loss": np.mean(all_losses)}, step=self.logging_epoch)

    def train(self):
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

        epoch_iterator = tqdm(
            range(self.num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0]
        )
        for epoch in epoch_iterator:  # range(1, self.num_epochs + 1):
            # if self.evaluate_during_training and self.local_rank in [-1, 0]:
                # torch.distributed.barrier() #! Uncomment those in case you want to try DDP
                # self.evaluate("develop") 
                # torch.distributed.barrier() #! Uncomment those in case you want to try DDP

            if isinstance(self.train_dataloader, DataLoader) and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.train_epoch()

            # if self.global_step > self.max_steps:  #! I have changed this
            #     print("Forced break because self.max_steps ({self.max_steps}) > self.global_step ({self.global_step}) 2")
            #     epoch_iterator.close()
            #     break
            self.logging_epoch += 1

        # #!Add  # Save the trained model and the tokenizer
        # if (self.local_rank == -1 or torch.distributed.get_rank() == 0):
        #     self.save_model_and_tokenizer()

        print(f"{'-'*100}\n...Done training!\n")
        self.model.eval()

        dataset_evaluated = None

        if self.local_rank in [-1, 0]:
            # self.model.save_model(save_dir=self.save_model_path, epoch=self.num_train_epochs)
            dataset_evaluated = self.evaluate("develop")
            print(f"{'-'*100}\n...Done evaluating!\n")

        return dataset_evaluated

    def evaluate(self, dataset_name="develop") -> Dict[str, float]:
        print(f"\n****** Running evaluation for: {dataset_name.upper()} ******\n")
        if dataset_name == "train":
            dataloader, dataset_featurized = self._get_dataloader(self.train_dataset, is_train=False)
            dataset = self.train_dataset.copy(deep=True)

        else:
            dataloader, dataset_featurized = self._get_dataloader(self.evaluate_dataset, is_train=False)
            dataset = self.evaluate_dataset.copy(deep=True)

        if self.verbose:
            # print(f"  Num examples for {website} = {len(data_loader.dataset)}")
            print(f"  Num examples for dataset = {len(dataset)}")
            print(f"  Batch size = {self.eval_batch_size}")

        self.model.eval()

        with torch.no_grad():
            all_logits = []
            all_losses = []
            batch_iterator = tqdm(dataloader, desc="Eval - Batch")
            for batch in batch_iterator:
                batch_list = self._batch_to_device(batch)
                inputs = {
                    "input_ids": batch_list[0],
                    "attention_mask": batch_list[1],
                    "token_type_ids": batch_list[2],
                    "labels": batch_list[3],  #? Removing this won't report loss
                }
                outputs = self.model(**inputs)
                loss = outputs["loss"]  # model outputs are always tuple in transformers (see doc)
                logits = outputs["logits"]  # which is (bs,seq_len,node_type)
                all_logits.append(logits.detach().cpu())

                if self.n_gpu > 1:
                    # #? to average on multi-gpu parallel (not distributed) training
                    loss = loss.mean()

                # # ! Not sure if I need this here to compute correctly the loss
                # if self.gradient_accumulation_steps > 1:
                #     loss = loss / self.gradient_accumulation_steps
                all_losses.append(loss.item())

        print(f"- Evaluation Loss: {np.mean(all_losses)}")
        if self.run:
            if self.just_evaluation:
                log_name = f"{dataset_name}_Evaluation_Loss_Final"
            else:
                log_name = f"{dataset_name}_Evaluation_Loss"
            self.run.log({log_name: np.mean(all_losses)})

        all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2) 

        node_probs = []
        for feature_index, feature_ids in enumerate(dataset_featurized.relative_first_tokens_node_index):
            node_probs.extend(all_probs[feature_index, [dataset_featurized.relative_first_tokens_node_index[feature_index]], 0][0])

        node_probs = np.array(node_probs)
        print(len(node_probs))
        print("dataset: ", len(dataset))
        dataset = dataset.explode('nodes').reset_index()
        dataset = dataset.join(pd.DataFrame(dataset.pop('nodes').tolist(), columns=["xpath","node_text","gt_tag","node_gt_text" ]))
        print(f"Memory: {sum(df.memory_usage(deep=True))/10**6:.2f} Mb")
        # df.drop(['html', "swde_features"], axis=1, inplace=True)
        dataset["html"] = dataset["html"].astype("category")
        print(f"Memory: {sum(dataset.memory_usage(deep=True))/10**6:.2f} Mb")
        dataset['node_prob'] = node_probs
        dataset['node_pred'] = node_probs > 0.5

        # TODO: move this out
        dataset["node_gt"] = dataset["gt_tag"] == 'PAST_CLIENT' 
        dataset["node_pred"] = dataset["node_pred"].apply(lambda x: "PAST_CLIENT" if x else "none")
        dataset["node_gt"] = dataset["node_gt"].apply(lambda x: "PAST_CLIENT" if x else "none")

        metrics_per_dataset, cm_per_dataset = self.get_classification_metrics(dataset)
        if self.run:
            self.log_metrics(metrics_per_dataset)
            self.log_metrics(cm_per_dataset)

        return dataset

    def log_metrics(self, metric):
        for key, value in metric.items():
            if self.just_evaluation:
                key = f"{key}_final"
            self.run.log({key: value}, step=self.logging_epoch)

    def get_classification_metrics(self, dataset_predicted):
        print("Compute Metrics:")
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(dataset_predicted)

        print(
            f"Node Classification Metrics per Dataset:\n {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )
        return metrics_per_dataset, cm_per_dataset


# %%
# from markuplmft.fine_tuning.run_swde.trainer import Trainer

print(f"\n{local_rank} - Preparing Trainer...")
# #? Leave this barrier here because it unlocks
# #? the other GPUs that were waiting at: 
# #? load_or_cache_websites in DataReader
if local_rank == 0: 
    torch.distributed.barrier()

trainer = Trainer(
    model = model,
    train_dataset = df,
    evaluate_dataset = df,
    featurizer = featurizer,
    local_rank=local_rank,
    device=device, 
    n_gpu=n_gpu,
    run=None,
    **trainer_config,
)

# %%
dd = trainer.train()

# %%
dd

# %%
dataset.relative_first_tokens_node_index[:10] #? Just to understand that the amout of nodes will match the number of nodes in the dataset
# #? In case we change this to multiple nodes that won't be the same and the absolute_node_index will be important to combine the signal of multiple features for one node
# dataset.absolute_node_index

# %%
# 0.6696927785873413
# 0.6806140422821045
