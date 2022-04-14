from __future__ import absolute_import, division, print_function

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
from markuplmft.fine_tuning.run_swde.utils import set_seed, get_device_and_gpu_count
from markuplmft.fine_tuning.run_swde import constants
from markuplmft.models.markuplm import (
    MarkupLMConfig,
    MarkupLMForTokenClassification,
    MarkupLMTokenizer,
)

logger = logging.getLogger(__name__)


class MarkupLModel:
    def __init__(
        self,
        pre_trained_model_folder_path: str = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/286",
        verbose: bool = False,
        no_cuda:bool = False
    ):
        self.net = None
        self.tokenizer = None

        self.local_rank = -1

        self.device, self.n_gpu = get_device_and_gpu_count(no_cuda, self.local_rank)
        set_seed(self.n_gpu)  # ? For reproducibility

        self.pre_trained_model_folder_path = pre_trained_model_folder_path

        # self.output_dir = Path("/data/GIT/unilm/markuplm/markuplmft/models/markuplm/")

        self.original_model_dir = Path("microsoft/markuplm-base")
        self.save_model_path = Path("/data/GIT/unilm/markuplm/markuplmft/models/markuplm/")

        self.doc_stride = 128
        self.max_seq_length = 384
        
        # self.overwrite_cache = False

        if verbose:
            print("self.__dict__", pprint(self.__dict__))

    def load_pretrained_model_and_tokenizer(self, local_rank):
        # ? Load pretrained model and tokenizer
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # ? Make sure only the first process in distributed training will download model & vocab
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.original_model_dir)

        config = MarkupLMConfig.from_pretrained(self.original_model_dir)
        config_dict = config.to_dict()
        config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE)})
        config = MarkupLMConfig.from_dict(config_dict)

        self.net = MarkupLMForTokenClassification.from_pretrained(
            self.original_model_dir, config=config
        )
        self.net.resize_token_embeddings(len(self.tokenizer))

    def load_trained_model(self, 
        config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/", 
        tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/", 
        net_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/1/checkpoint-1"):

        config = MarkupLMConfig.from_pretrained(config_path)
        self.tokenizer = MarkupLMTokenizer.from_pretrained(tokenizer_path)
        self.net = MarkupLMForTokenClassification.from_pretrained(net_path, config=config)
        self.net.to(self.device)

    def save_model_and_tokenizer(self):
        # TODO (aimore): Replace os with Path [done - remove comment if passes]
        # if not os.path.exists(self.save_model_path) and self.local_rank in [-1, 0]: 
        #     os.makedirs(self.save_model_path)
        if self.local_rank in [-1, 0]: 
            self.save_model_path.mkdir(parents=True, exist_ok=False) #! Uncomment

        print(f"Saving model checkpoint to {self.save_model_path}")
        #? Save a trained model, configuration and tokenizer using `save_pretrained()`.
        #? They can then be reloaded using `from_pretrained()`
        #? Take care of distributed/parallel training
        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        model_to_save.save_pretrained(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)
        # TODO(aimore): Save the parameters: torch.save(self.__dict__, os.path.join(self.save_model_path, "training_args.bin"))

    def save_model(self, save_dir, epoch):
        self.save_path = Path(save_dir) / f"checkpoint-{epoch}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        #? Take care of distributed/parallel training
        model_to_save.save_pretrained(self.save_path)
        #! Save the parameters: torch.save(self.__dict__, os.path.join(output_dir, "training_args.bin"))
        if self.save_path.exists():
            print(f"Overwriting model checkpoint: {self.save_path}")
        else:
            print(f"Saving model checkpoint to: {self.save_path}")

