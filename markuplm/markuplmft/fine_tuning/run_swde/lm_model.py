from __future__ import absolute_import, division, print_function
import multiprocessing as mp
import argparse
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

seed = 66 #! Set the seed like other modules


class LModel():
    def __init__(self):
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

        self.data_root_dir = "/data/GIT/swde/my_data/train/my_CF_processed/"

        self.doc_stride = 128
        self.max_seq_length = 384
        self.overwrite_cache = False
        self.save_features = True

        self.do_train = True
        self.parallelize = True
        self.max_steps = -1

        #? Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count() if not self.no_cuda else 0
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        #? Prepare pre-trained tokenizer
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.pre_trained_model_folder_path)

    def load_data(self):
        print(f"Loading data from: {self.data_root_dir}")
        swde_path = Path(self.data_root_dir)
        websites = [
            file_path.parts[-1]
            for file_path in list(swde_path.iterdir())
            if "cached" not in str(file_path)
        ]
        websites = [website for website in websites if "ciphr.com" not in website] #! Remove this website for now just because it is taking too long (+20min.) 
        websites = websites[:10] #! Just for speed reasons
        print(f"Websites ({len(websites)}): {websites}")
        train_websites = websites

        logger.info(f"\nWebsites ({len(websites)}):\n{websites}\n")

        # first we load the features
        global feature_dicts
        feature_dicts = self.load_or_cache_websites(websites=websites)

        # global global_feature_dicts
        # global_feature_dicts = feature_dicts

        if self.do_train:
            self.train_dataset, _ = self.get_dataset_and_info_for_websites(train_websites)
        else:
            self.eval_dataset, self.info = self.get_dataset_and_info_for_websites(eval_websites, evaluate-True)

    def load_or_cache_websites(self, websites:List[str]) -> Dict:
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        feature_dicts = {}

        if self.parallelize:
            num_cores = mp.cpu_count()
            with mp.Pool(num_cores) as pool, tqdm(total=len(websites), desc="Loading/Creating Features") as t:
                for website, features_per_website in pool.imap_unordered(self.load_or_cache_one_website_features, websites):
                    feature_dicts[website] = features_per_website
        else:
            for website in websites:
                features_per_website = self.load_or_cache_one_website_features(website)
                feature_dicts[website] = features_per_website

        print(f"Features size: {len(feature_dicts)}")
        return feature_dicts

    def load_or_cache_one_website_features(self, website):
        cached_features_file = os.path.join(
            self.data_root_dir,
            "cached",
            website,
            f"cached_markuplm_{str(self.max_seq_length)}",
        )

        if not os.path.exists(os.path.dirname(cached_features_file)):
            os.makedirs(os.path.dirname(cached_features_file))

        if os.path.exists(cached_features_file) and not self.overwrite_cache:
            logger.info(f"Loading features from cached file: {cached_features_file}")
            features = torch.load(cached_features_file)

        else:
            logger.info(f"Creating features for: {website}")

            features = get_swde_features(
                root_dir=self.data_root_dir,
                website=website,
                tokenizer=self.tokenizer,
                doc_stride=self.doc_stride,
                max_length=self.max_seq_length,
            )

            if self.local_rank in [-1, 0] and self.save_features:
                logger.info(f"Saving features into cached file: {cached_features_file}\n")
                torch.save(features, cached_features_file)

        return website, features

    def get_dataset_and_info_for_websites(self, websites: List, evaluate=False):
        print("Getting data information for websites: ")
        all_features = []

        for website in websites:
            features_per_website = feature_dicts[website]
            all_features += features_per_website

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in all_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in all_features], dtype=torch.long)
        all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in all_features], dtype=torch.long)
        all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in all_features], dtype=torch.long)

        if not evaluate:
            all_labels = torch.tensor([f.labels for f in all_features], dtype=torch.long)
            dataset = SwdeDataset(
                all_input_ids=all_input_ids,
                all_attention_mask=all_attention_mask,
                all_token_type_ids=all_token_type_ids,
                all_xpath_tags_seq=all_xpath_tags_seq,
                all_xpath_subs_seq=all_xpath_subs_seq,
                all_labels=all_labels,
            )
            info = None
        else:
            # in evaluation, we do not add labels
            dataset = SwdeDataset(
                all_input_ids=all_input_ids,
                all_attention_mask=all_attention_mask, 
                all_token_type_ids=all_token_type_ids,
                all_xpath_tags_seq=all_xpath_tags_seq,
                all_xpath_subs_seq=all_xpath_subs_seq,
            )
            info = [
                (
                    f.html_path, #? '1820productions.com.pickle-0000.htm'
                    f.involved_first_tokens_pos, #? [1, 1, 34, 70, 80]
                    f.involved_first_tokens_xpaths, #? ['/html/head', '/html/head/script[1]', '/html/head/script[2]', '/html/head/title', '/html/head/script[3]']
                    f.involved_first_tokens_types, #? ['none', 'none', 'none', 'none', 'none']
                    f.involved_first_tokens_text, #? ['', "var siteConf = { ajax_url: 'https://1820productions.com/wp-admin/admin-ajax.php' };", "(function(html){html.className = html.c ......."]
                )
                for f in all_features
            ]
        print("... Done!")
        return dataset, info #? This info is used for for evaluation (store the groundtruth) 

    def prepare_model_to_train(self):        
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and not self.overwrite_output_dir
        ):
            raise ValueError(f"Output directory ({self.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

        # Set seed
        set_seed(self.n_gpu)

        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Make sure only the first process in distributed training will download model & vocab

        #? Prepare pre-trained model
        config = MarkupLMConfig.from_pretrained(self.pre_trained_model_folder_path)
        config_dict = config.to_dict()
        config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE)})
        config = MarkupLMConfig.from_dict(config_dict)

        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.model_name_or_path)

        self.model = MarkupLMForTokenClassification.from_pretrained(self.model_name_or_path, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        # self.tokenizer.save_pretrained(sub_output_dir)

        #? Prepare training data
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        train_sampler = RandomSampler(self.train_dataset) if self.local_rank == -1 else DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size,
        )

        #? Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param for param_name, param in self.model.named_parameters() if not any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param for param_name, param in self.model.named_parameters() if any(nd in param_name for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)

        if self.max_steps > 0:
            self.t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps // (len(self.train_dataloader) // self.gradient_accumulation_steps) + 1
            )
        else:
            self.t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs


        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(self.warmup_ratio * self.t_total), 
            num_training_steps=self.t_total
        )
        #? Set model to use fp16 and multi-gpu
        if self.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )


    def fit(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {self.per_gpu_train_batch_size}")

        total_train_batch_size = self.train_batch_size * self.gradient_accumulation_steps * (torch.distributed.get_world_size() if self.local_rank != -1 else 1)
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.t_total}")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            int(self.num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0]
        )
        set_seed(self.n_gpu)  # Added here for reproductibility (even between python 2 and 3)
        for epoch in train_iterator:
            if isinstance(self.train_dataloader, DataLoader) and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
            for step, batch_list in enumerate(epoch_iterator):
                self.model.train()
                batch_list = tuple(batch.to(self.device) for batch in batch_list)
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

                if self.fp16: # TODO (Aimore): Replace this with Accelerate
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.local_rank in [-1, 0]
                        and self.logging_steps > 0
                        and global_step % self.logging_steps == 0
                    ):
                        # Log metrics
                        if self.local_rank == -1 and self.evaluate_during_training:
                            raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")
                            # results = evaluate(args, model, tokenizer, prefix=str(global_step))
                            # for key, value in results.items():
                            #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        #! tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        #! tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                        logging_loss = tr_loss

                    if (
                        self.local_rank in [-1, 0]
                        and self.save_steps > 0
                        and global_step % self.save_steps == 0
                    ):
                        # Save model checkpoint
                        self.save_model(sub_output_dir, global_step)

                if 0 < self.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.max_steps < global_step:
                train_iterator.close()
                break

        # if self.local_rank in [-1, 0]:
        #     tb_writer.close()

        # Save the trained model and the tokenizer
        if self.do_train and (self.local_rank == -1 or torch.distributed.get_rank() == 0):
            self.save_model_and_tokenizer()
        
        return global_step, tr_loss / global_step

    def save_model_and_tokenizer(self):
        # Create output directory if needed
        if not os.path.exists(self.save_model_path) and self.local_rank in [-1, 0]:
            os.makedirs(self.save_model_path)

        logger.info(f"Saving model checkpoint to {self.save_model_path}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)
        torch.save(self.__dict__, os.path.join(self.save_model_path, "training_args.bin"))        

    @staticmethod
    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    # def predict(self):
        # set_seed(self.n_gpu)
        # model = MarkupLMForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
        # model.resize_token_embeddings(len(self.tokenizer))

    def save_model(self, sub_output_dir, global_step):
        output_dir = os.path.join(sub_output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(self.__dict__, os.path.join(output_dir, "training_args.bin"))
        logger.info(f"Saving model checkpoint to {output_dir}")

    def load_model(self):
        pass
