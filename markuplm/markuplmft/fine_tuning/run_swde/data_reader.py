from __future__ import absolute_import, division, print_function
import multiprocessing as mp

# import copy
from pprint import pprint
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch

# from markuplmft.fine_tuning.run_swde.eval_utils import page_level_constraint
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.data_feature_utils import SwdeDataset, get_swde_features

from markuplmft.fine_tuning.run_swde.utils import set_seed
from markuplmft.models.markuplm import (
    MarkupLMTokenizer,
)

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )

logger = logging.getLogger(__name__)


DOC_STRIDE = 128
MAX_SEQ_LENGTH = 384


class DataReader:
    def __init__(self, local_rank, n_gpu, **config):
        self.local_rank = local_rank

        # set_seed(n_gpu=n_gpu)

        self.tokenizer_dir = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/286"

        self.doc_stride = DOC_STRIDE
        self.max_seq_length = MAX_SEQ_LENGTH
        self.overwrite_cache = config["overwrite_cache"]

        self.parallelize = config["parallelize"]
        self.verbose = config["verbose"]

        # ? Prepare pre-trained tokenizer
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.tokenizer_dir)
        if self.verbose:
            print("self.__dict__", pprint(self.__dict__))

    def load_dataset(
        self,
        data_dir="/data/GIT/swde/my_data/train/my_CF_processed/",
        limit_data=False,
    ):
        self.data_dir = Path(data_dir)
        print(f"Loading data from: {self.data_dir}")
        # ? Look for the existing websites in the folder
        websites = [
            file_path.parts[-1]
            for file_path in list(self.data_dir.iterdir())
            if "cached" not in str(file_path)
        ]

        stop_websites = {
            "ciphr.com.pickle",
            "hawthorneadvertising.com.pickle",
            "eng.it.pickle",
        }  #! Remove this website for now just because it is taking too long (+20min.)
        self.websites = sorted(list(set(websites) - stop_websites))

        if limit_data:
            # self.websites = self.websites[:limit_data]  #! Just for speed reasons
            self.websites = self.websites[limit_data:]

        print(f"\nWebsites ({len(self.websites)}):\n{self.websites}\n")

        # ? Load all features for websites
        global feature_dicts
        feature_dicts = self.load_or_cache_websites(websites=self.websites)
        print("...Loaded/Created feature_dicts Done!")

        dataset, info = self.get_dataset_and_info_for_websites(self.websites)
        return dataset, info

    def load_or_cache_websites(self, websites: List[str]) -> Dict:
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        feature_dicts = {}

        if self.parallelize:
            num_cores = mp.cpu_count()
            print(f"Parallelizing on num_cores: {num_cores}")
            # with mp.Pool(num_cores) as pool, tqdm(total=len(websites), disable=self.local_rank not in [-1, 0]) as t:
            with mp.Pool(num_cores) as pool:
                for website, features_per_website in pool.imap(
                    self.load_or_cache_one_website_features, websites
                ):
                    feature_dicts[website] = features_per_website
                    # t.update()
                    # t.set_description(f"Loading/Creating Features ({website})")
        else:
            for website in tqdm(websites, disable=self.local_rank not in [-1, 0]):
                feature_dicts[website] = self.load_or_cache_one_website_features(website)

        if self.local_rank == 0: # !Remove
            torch.distributed.barrier()

        return feature_dicts

    def load_or_cache_one_website_features(self, website):
        cached_features_dir = (
            self.data_dir / "cached" / f"cached_markuplm_{str(self.max_seq_length)}"
        )
        cached_features_file = cached_features_dir / website

        if not cached_features_dir.exists():
            if self.verbose:
                print(f"Creating cache folder: {cached_features_file}")
            cached_features_dir.mkdir(exist_ok=False, parents=True)

        if cached_features_file.exists() and not self.overwrite_cache:
            if self.verbose:
                print(f"Loading features from cached file: {cached_features_file}")
            features = torch.load(str(cached_features_file))
        else:
            if self.verbose:
                print(f"Creating features: {cached_features_file}")

            features = get_swde_features(
                root_dir=self.data_dir,
                website=website,
                tokenizer=self.tokenizer,
                doc_stride=self.doc_stride,
                max_length=self.max_seq_length,
            )

            if self.local_rank in [-1, 0]:
                if self.verbose:
                    print(f"Saving features into cached file: {cached_features_file}\n")
                torch.save(features, str(cached_features_file))

        if self.parallelize:
            return website, features
        else:
            return features


    def get_dataset_and_info_for_websites(self, websites: List):
        print("Getting data information for websites: ")
        all_features = []

        for website in websites:
            if self.verbose:
                print(website)
            features_per_website = feature_dicts[website]
            all_features += features_per_website

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in all_features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in all_features], dtype=torch.long
        )
        all_xpath_tags_seq = torch.tensor(
            [f.xpath_tags_seq for f in all_features], dtype=torch.long
        )
        all_xpath_subs_seq = torch.tensor(
            [f.xpath_subs_seq for f in all_features], dtype=torch.long
        )

        
        # all_xpath_subs_seq

        #! Removed the evaluation from here so that all datasets have all_labels (loss) and info
        all_labels = torch.tensor([f.labels for f in all_features], dtype=torch.long)
        dataset = SwdeDataset(
            all_input_ids=all_input_ids,
            all_attention_mask=all_attention_mask,
            all_token_type_ids=all_token_type_ids,
            all_xpath_tags_seq=all_xpath_tags_seq,
            all_xpath_subs_seq=all_xpath_subs_seq,
            all_labels=all_labels,
        )
        info = [
            (
                f.html_path,  # ? '1820productions.com.pickle-0000.htm'
                f.involved_first_tokens_pos,  # ? [1, 1, 34, 70, 80]
                f.involved_first_tokens_xpaths,  # ? ['/html/head', '/html/head/script[1]', '/html/head/script[2]', '/html/head/title', '/html/head/script[3]']
                f.involved_first_tokens_types,  # ? ['none', 'none', 'none', 'none', 'none']
                f.involved_first_tokens_text,  # ? ['', "var siteConf = { ajax_url: 'https://1820productions.com/wp-admin/admin-ajax.php' };", "(function(html){html.className = html.c ......."]
                f.involved_first_tokens_gt_text,  # ?
                f.involved_first_tokens_node_attribute,  # ?
                f.involved_first_tokens_node_tag,  # ?
            )
            for f in all_features
        ]

        print("... Done!")
        return dataset, info  # ? This info is used for evaluation (store the groundtruth)


    # def get_dataset_and_info_for_website(self, features):
    #     # Convert to Tensors and build dataset
    #     all_input_ids = torch.tensor(features.input_ids, dtype=torch.long)
    #     all_attention_mask = torch.tensor(features.attention_mask, dtype=torch.long)
    #     all_token_type_ids = torch.tensor(features.token_type_ids, dtype=torch.long)
    #     all_xpath_tags_seq = torch.tensor(features.xpath_tags_seq, dtype=torch.long)
    #     all_xpath_subs_seq = torch.tensor(features.xpath_subs_seq, dtype=torch.long)

    #     #! Removed the evaluation from here so that all datasets have all_labels (loss) and info
    #     all_labels = torch.tensor(features.labels, dtype=torch.long)
    #     one_dataset = SwdeDataset(
    #         all_input_ids=all_input_ids,
    #         all_attention_mask=all_attention_mask,
    #         all_token_type_ids=all_token_type_ids,
    #         all_xpath_tags_seq=all_xpath_tags_seq,
    #         all_xpath_subs_seq=all_xpath_subs_seq,
    #         all_labels=all_labels,
    #     )
    #     one_info = (
    #         features.html_path,  # ? '1820productions.com.pickle-0000.htm'
    #         features.involved_first_tokens_pos,  # ? [1, 1, 34, 70, 80]
    #         features.involved_first_tokens_xpaths,  # ? ['/html/head', '/html/head/script[1]', '/html/head/script[2]', '/html/head/title', '/html/head/script[3]']
    #         features.involved_first_tokens_types,  # ? ['none', 'none', 'none', 'none', 'none']
    #         features.involved_first_tokens_text,  # ? ['', "var siteConf = { ajax_url: 'https://1820productions.com/wp-admin/admin-ajax.php' };", "(function(html){html.className = html.c ......."]
    #         features.involved_first_tokens_gt_text,  # ?
    #     )
    #     return one_dataset, one_info

    # def get_dataset_and_info_for_websites(self, websites: List):
    #     if self.verbose:
    #         print("Getting data information for websites: ")

    #     all_features = []
    #     for website in tqdm(websites, leave=False):
    #         if self.verbose:
    #             print(website)
    #         features_per_website = feature_dicts[website]
    #         all_features += features_per_website

    #     all_datasets, all_infos = np.array(
    #         [
    #             self.get_dataset_and_info_for_website(features)
    #             for features in tqdm(all_features, total=len(all_features))
    #         ]
    #     ).T

    #     print("... Done!")
    #     return all_datasets, all_infos  # ? This info is used for evaluation (store the groundtruth)
