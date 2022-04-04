from __future__ import absolute_import, division, print_function
import multiprocessing as mp
import argparse
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset

# import copy
from pprint import pprint
import collections
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

logger = logging.getLogger(__name__)


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


seed = 66  #! Set the seed like other modules


class MarkupLModel:
    def __init__(
        self,
        pre_trained_model_folder_path: str = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/286",
    ):
        self.model = None
        self.tokenizer = None
        self.no_cuda = False
        self.local_rank = -1

        self.pre_trained_model_folder_path = pre_trained_model_folder_path

        self.output_dir = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/"
        self.overwrite_output_dir = True

        self.model_name_or_path = "microsoft/markuplm-base"
        self.save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/"

        self.doc_stride = 128
        self.max_seq_length = 384
        self.overwrite_cache = False
        self.save_features = True

        # ? Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
            )
            self.n_gpu = torch.cuda.device_count() if not self.no_cuda else 0
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        print("self.__dict__", pprint(self.__dict__))

    def load_pretrained_tokenizer():
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.model_name_or_path)

    def load_pretrained_model(self):

        # Set seed
        set_seed(self.n_gpu)

        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Make sure only the first process in distributed training will download model & vocab

        config = MarkupLMConfig.from_pretrained(self.pre_trained_model_folder_path)
        config_dict = config.to_dict()
        config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE)})
        config = MarkupLMConfig.from_dict(config_dict)


        self.model = MarkupLMForTokenClassification.from_pretrained(
            self.model_name_or_path, config=config
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        # self.tokenizer.save_pretrained(sub_output_dir)

    def save_model_and_tokenizer(self):
        # Create output directory if needed
        if not os.path.exists(self.save_model_path) and self.local_rank in [-1, 0]:
            os.makedirs(self.save_model_path)

        print(f"Saving model checkpoint to {self.save_model_path}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)
        #! Save the parameters: torch.save(self.__dict__, os.path.join(self.save_model_path, "training_args.bin"))

    def predict_on_develop(self):
        set_seed(self.n_gpu)
        self.load_model()

        dataset_predicted = pd.DataFrame()
        for website in tqdm(self.websites):
            website_predicted = self.predict_on_website(website)
            website_predicted["domain"] = website
            dataset_predicted = dataset_predicted.append(website_predicted)

        return dataset_predicted

    def predict_on_website(self, website):
        dataset, info = self.get_dataset_and_info_for_websites([website], evaluate=True)

        eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        # Note that DistributedSampler samples randomly
        # In our setting, we should not apply DDP
        eval_sampler = (
            SequentialSampler(dataset) if self.local_rank == -1 else DistributedSampler(dataset)
        )

        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info(f"***** Running evaluation *****")
        logger.info(f"  Num examples for {website} = {len(dataset)}")
        logger.info(f"  Batch size = {eval_batch_size}")

        all_logits = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(
                b.to(self.device) for b in batch
            )  # TODO (AIMORE): Why they use tuple here?
            with torch.no_grad():
                inputs = {  # TODO (AIMORE): Can't this batch have better names, instead of these numbered indices?
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "xpath_tags_seq": batch[3],
                    "xpath_subs_seq": batch[4],
                }
                outputs = self.model(**inputs)
                logits = outputs["logits"]  # which is (bs,seq_len,node_type)
                all_logits.append(logits.detach().cpu())

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

    def evaluate(self, dataset_predicted):
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(dataset_predicted)
        print(
            f"Node Classification Metrics per Dataset: {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )

    def save_model(self, sub_output_dir, global_step):
        output_dir = os.path.join(sub_output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        #! Save the parameters: torch.save(self.__dict__, os.path.join(output_dir, "training_args.bin"))
        print(f"Saving model checkpoint to {output_dir}")

    def load_model(self):
        config = MarkupLMConfig.from_pretrained(self.pre_trained_model_folder_path)
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.pre_trained_model_folder_path)
        self.model = MarkupLMForTokenClassification.from_pretrained(
            self.pre_trained_model_folder_path, config=config
        )
        self.model.to(self.device)
