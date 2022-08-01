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
# # Check for the dataset format
# 1. This notebook checks if the format of the data and groundtruth is ok.
# 2. It does some stats on the 'none' and 'PAST_CLIENT' nodes 

# %%
import pandas as pd
df = pd.read_pickle("/data/GIT/node_classifier_with_imgs/develop/processed_dedup.pkl")

# %%
from markuplmft.fine_tuning.run_swde.prepare_everything import PrepareData
from markuplmft.fine_tuning.run_swde.prepare_everything import LabelHandler
dataset_name = 'develop'
wae_data_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
raw_data_folder = Path(f"/data/GIT/node_classifier_with_imgs/{dataset_name}")
prep_data = PrepareData(parallel=False, remove_folder_flag=False, shortcut=True, raw_data_folder=raw_data_folder)
all_df = prep_data.load_data(wae_data_path)


# %%
lh = LabelHandler()
all_df = lh.format_annotation(all_df)


# %%
399 / 429

# %%
all_df.groupby("domain")['PAST_CLIENT-gt_value'].agg(list).explode().value_counts() # 429

# %%
pd.set_option('display.max_columns',10, 'display.max_colwidth', 100, 'display.max_rows',10, 'display.min_rows',10)

pd.DataFrame(df[df["PAST_CLIENT-gt_text_count"]>0])

# df[df["PAST_CLIENT-gt_text_count"]>0]["PAST_CLIENT-gt_text"]
# for enum, x in enumerate(df[df["PAST_CLIENT-gt_text_count"]>0]["PAST_CLIENT-gt_text"].values):
#     print(enum, x)

# %%
pd.set_option('display.max_columns',20, 'display.max_colwidth', 50, 'display.max_rows',20, 'display.min_rows',20)
df.iloc[:10]

# %%
print(len(df_nodes))

# %%

df.groupby("domain")['PAST_CLIENT-gt_value'].agg(list).explode().value_counts() # 399
# 399 past clients

# %%
# df_nodes = df.explode("nodes").reset_index()
# df_nodes = df_nodes.join(
#     pd.DataFrame(
#         df_nodes.pop("nodes").tolist(),
#         columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
#     )
# )

pd.set_option('display.max_columns',20, 'display.max_colwidth', 50, 'display.max_rows',20, 'display.min_rows',20)

t = df_nodes[df_nodes["node_gt_tag"] != 'none'].apply(lambda x: [y for y in x["node_gt_text"] if 'http' in y], axis=1)
df_nodes.loc[t[t.apply(len)==0].index]

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',200, 'display.min_rows',200)

z = df[df["PAST_CLIENT-gt_text_count"]>0].explode('nodes')
# z[z[2] !='none']
z

# %% tags=[]
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
pd.set_option("min_rows",5, "max_rows", 5)

# %% tags=[]
# dataset = 'train'
dataset = 'develop'
dedup = True
if dedup:
    dedup = "_dedup"
else:
    dedup = ""

# %% [markdown]
# # Packed Data (Data after pack_data.py)

# %% tags=[]
data_path = f'../../../swde/my_data/{dataset}/my_CF_sourceCode/wae.pickle'
data_packed = pd.read_pickle(data_path)
len(data_packed)

# %% [markdown]
# ### Ground Truth

# %% tags=[]
gt_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_sourceCode/groundtruth/'

for gt_file in list(gt_path.iterdir())[:]:
    print(gt_file)
    with open(gt_file) as text:
        lines = text.readlines()
        for l in lines:
            print(l)

# %% [markdown]
# # Prepare Data (Data after prepare_data.py)
#

# %% [markdown] tags=[]
# ## Check if all websites have at least one tag

# %% tags=[]
pd.set_option('max_colwidth', 2000)

websites_root_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_processed{dedup}/'
print(f"Loading data from: {websites_root_path}")
websites_data_path = list(websites_root_path.glob('*'))
websites_data_path = [website_path for website_path in websites_data_path if 'cache' not in str(website_path)]
print(len(websites_data_path))

# %%
websites_data_path

# %%
if dataset == 'train':
    assert len(websites_data_path) == len(data_packed), f"{len(websites_data_path)} != {len(data_packed)}"
else:
    assert len(websites_data_path) == len(data_packed) - 1, f"{len(websites_data_path)} != {len(data_packed)}"

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

import multiprocess as mp

p = mp.Pool(mp.cpu_count())
all_dfs = pd.DataFrame()

for dfs in p.imap(read_data, websites_data_path):
    all_dfs = all_dfs.append(dfs)

len(all_dfs)

# %%
all_dfs['text_len'] = all_dfs['text'].apply(lambda  x: len(x.strip()))
all_dfs['gt_text_count'] = all_dfs['gt_text'].apply(len)

# %%
all_dfs[all_dfs["xpath"] == '/html/head/title[1]'].sort_values("gt_field")

# %% [markdown]
# # Label Analysis

# %% [markdown]
# ## Stats

# %%
all_dfs["gt_text_count"].sum()

# %% [markdown]
# ## Duplicated node_text 

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
# ## _node_tag_ and _node_attribute_
# By identified where it is not likely a positive label to appear we should remove those cases and limit the scope of the data in order to:
# - Reduce time in all stages 
# - Remove noisy data

# %%
all_dfs[all_dfs["node_tag"] == 'title']

# %%
print(all_dfs.columns.values)


# %%
def node_analysis(all_dfs, col:str):
    all_data_size = len(all_dfs)
    positives = all_dfs[all_dfs["gt_text_count"] > 0]
    negatives = all_dfs[all_dfs["gt_text_count"] == 0]

    positives_col = positives[col].value_counts()
    negatives_col = negatives[col].value_counts()
    diff_col_set = set(negatives_col.index) - set(positives_col.index)
    print(f"positives_col: {len(positives_col)}")
    print(f"negatives_col: {len(negatives_col)}")
    print(f"diff_col_set: {len(diff_col_set)}")
    negatives_col.loc[list(diff_col_set)]

    diff_nodes = all_dfs[all_dfs[col].isin(diff_col_set)]
    print(f"Number of nodes containing this difference: {len(diff_nodes)} ({100*len(diff_nodes)/all_data_size:.2f}% of the total)")
    print(f"% of the memory that represents: {diff_nodes['text_len'].sum()} ({100*diff_nodes['text_len'].sum()/all_dfs['text_len'].sum():.2f}% of the total)")

    distribution = pd.DataFrame(positives_col).join(pd.DataFrame(negatives_col),how="outer",  lsuffix='_pos', rsuffix='_neg')
    with pd.option_context("min_rows", 30, "max_rows", 30):
        print(distribution.sort_values(f"{col}_neg", ascending=False))


# %%
node_analysis(all_dfs, "node_tag")

# %%
node_analysis(all_dfs, "node_attribute")

# %%
all_dfs["node_attribute"].value_counts()

# %% [markdown]
# ## Understand the difference between the positive and negative node text length distribution

# %%
print("Positive node text length distribution:")
pd.DataFrame(all_dfs[all_dfs['gt_text_count']>0]['text_len'].describe()).style.format(na_rep='MISS', precision=1)  

# %%
print("Negative node text length distribution:")
pd.DataFrame(all_dfs[all_dfs['gt_text_count']==0]['text_len'].describe()).style.format(na_rep='MISS', precision=1)  

# %% [markdown]
# ## Check how long the node text can be in order to remove high memorydata
#

# %%
max_len = 10000
all_gt = all_dfs["gt_text_count"].sum()
gt_numb = all_dfs[all_dfs["text_len"] > max_len]["gt_text_count"].sum()
data_numb = len(all_dfs[all_dfs["text_len"] > max_len]) 
all_data = len(all_dfs)

text_len = all_dfs[all_dfs["text_len"] > max_len]["text_len"].sum()
all_text_len = all_dfs["text_len"].sum()

print(f"gt_numb: {gt_numb}")
print(f"data_numb: {data_numb}")
print(f"ratio:\n gt_numb_ratio: {100*gt_numb/all_gt:.2f} %\n data_numb_ratio: {100*data_numb/all_data:.2f} %\n text_len_ratio: {100*text_len/all_text_len:.2f} %")

# %% [markdown]
# ---
