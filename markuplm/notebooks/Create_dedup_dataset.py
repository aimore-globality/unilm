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

# %% [markdown]
# # Create deduplicated version of the Datasets

# %% tags=[]
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import multiprocess as mp
import os 
os.environ["WANDB_NOTEBOOK_NAME"] = "Create_dedup_datasaet.ipynb"
import wandb

pd.set_option("min_rows",5, "max_rows", 5)
wandb.login()

run = wandb.init(project="LanguageModel", resume=False, tags=["create_dedup"])

# %% tags=[]
dataset = 'train'
# dataset = 'develop'
print("Dataset: ", dataset)

# %% [markdown]
# ## Load Prepare Data (prepare_data.py)

# %% tags=[]
websites_root_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_processed/'
print(f"Loading data from: {websites_root_path}")
websites_data_path = list(websites_root_path.glob('*'))
websites_data_path = [website_path for website_path in websites_data_path if 'cache' not in str(website_path)]
print(len(websites_data_path))

# %%
websites_data_path

# %%
print(len(websites_data_path))

def read_data(website_path):
    dfs = pd.DataFrame()
    website_data = pd.read_pickle(website_path)
    for page_index in website_data.keys():
        website = str(website_path.parts[-1]).split('.pickle')[0]
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'gt_field', 'gt_text', 'node_attribute', 'node_tag'])
        df["page_index"] = page_index
        if len(df) > 0:
            df["website"] = website
            dfs = dfs.append(df)
    return dfs

p = mp.Pool(mp.cpu_count())
all_dfs = pd.DataFrame()

for dfs in p.imap(read_data, websites_data_path):
    all_dfs = all_dfs.append(dfs)

len(all_dfs)

# %%
all_dfs['text_len'] = all_dfs['text'].apply(lambda  x: len(x.strip()))
all_dfs['gt_text_len'] = all_dfs['gt_text'].apply(len)

# %%

# #? Interesting analysis if we remove the nodes with duplicated data, we can massively reduce their size.
duplicated_nodes = all_dfs
domain_deduplicated_nodes = all_dfs.drop_duplicates(subset=["text", "website"])
print(f"{'All nodes:':>50} {len(duplicated_nodes):>7}")
print(f"{'Domain deduplicated nodes:':>50} {len(domain_deduplicated_nodes):>7} ({100*len(domain_deduplicated_nodes)/len(duplicated_nodes):.2f} %)")

# #? Also, not so many nodes with positive data are removed compared to the other data.
duplicated_gt = len(duplicated_nodes[duplicated_nodes["gt_field"] != 'none'])
domain_deduplicated_gt = len(domain_deduplicated_nodes[domain_deduplicated_nodes["gt_field"] != 'none'])
print(f"{'All number of ground truth nodes:':>50} {duplicated_gt:>7}")
print(f"{'Domain deduplicated ground truth nodes:':>50} {domain_deduplicated_gt:>7} ({100*(domain_deduplicated_gt) / duplicated_gt:.2f} %)")

# %% [markdown]
# ## Create a nonduplicated data
# For the training: Keep the duplicated ground truth node 

# %%
print(len(websites_data_path))

def create_domain_deduplicated_data(folder, domain_deduplicated_nodes):
    print("Creating dedup node dataset:")
    folder = Path(str(websites_root_path) + "_dedup")
    if not folder.exists():
        folder.mkdir()

    for website, website_data in domain_deduplicated_nodes.groupby("website"):
        d = dict()
        save_path = folder / (str(website) + ".pickle")
        for page_index, page_data in website_data.groupby("page_index"):            
            d[page_index] = [tuple(x) for x in page_data[['text', 'xpath', 'gt_field', 'gt_text', 'node_attribute', 'node_tag']].values]

        print(f"save_path: {save_path}")
        pd.to_pickle(d, save_path)

if  dataset == 'train': #? If the data is for training keep the duplicated gt, but for develop/inference drop all duplicates 
    all_dfs = all_dfs.reset_index()
    indices = set(all_dfs.drop_duplicates(subset=["text", "website"]).index).union(set(duplicated_nodes[duplicated_nodes["gt_field"] != 'none'].index))
    domain_deduplicated_nodes = all_dfs.loc[indices]

create_domain_deduplicated_data(folder=websites_root_path, domain_deduplicated_nodes=domain_deduplicated_nodes)

# %%
run.save()
run.finish()

# %% [markdown]
# ---
