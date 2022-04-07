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
#     display_name: markuplmft
#     language: python
#     name: python3
# ---

# %% tags=[]
from markuplmft.fine_tuning.run_swde.data_reader import DataReader

config = dict(
    overwrite_cache=False,
    parallelize=False, 
    verbose=False)
dr = DataReader(**config)

# #! Check if the node has a positive, otherwise 

train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=4, min_node_text_size=20)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=4, min_node_text_size=2)

# #?  I will use 24 websites to train and 8 websites to evaluate
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=24)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=8)

# #?  Generate all features
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=False)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=False)

# %%
train_dataset_info[0]

# %%
import pandas as pd
from tqdm import tqdm

batch = pd.DataFrame()
for batch_list in tqdm(train_dataset_info[1]):
    inputs = {
        "input_ids": batch_list[0],
        "attention_mask": batch_list[1],
        "token_type_ids": batch_list[2],
        "xpath_tags_seq": batch_list[3],
        "xpath_subs_seq": batch_list[4],
        "labels": batch_list[5],
    }
    batch = batch.append(pd.DataFrame(inputs))
len(batch)

# %%
batch["xpath_subs_seq_len"] = batch["xpath_subs_seq"].apply(len)

# %%
batch

# %%
batch

# %%
