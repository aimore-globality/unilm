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

# %% [markdown]
# # Check for the dataset format
# 1. This notebook checks if the format of the data and groundtruth is ok.
# 2. It does some stats on the 'none' and 'PAST_CLIENT' nodes 

# %%

# %% tags=[]
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
pd.set_option("min_rows",5, "max_rows", 5)

# %% tags=[]
dataset = 'train'

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

websites_root_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_processed/'
print(websites_root_path)
websites_data_path = list(websites_root_path.glob('*'))
websites_data_path = [website_path for website_path in websites_data_path if 'cache' not in str(website_path)]
print(len(websites_data_path))
websites_data_path

# %%
assert len(websites_data_path) == len(data_packed), f"{len(websites_data_path)} != {len(data_packed)}"

# %% tags=[]
all_dfs = {}
for website_path in tqdm(websites_data_path):
    dfs = []
    website_data = pd.read_pickle(website_path)
    break
    # print(f"{website_path} {len(website_data)}")
    # # if website_path == '/data/GIT/swde/my_data/develop/my_CF_processed/ciphr.com.pickle':
    # for page_index in website_data.keys():
    #     df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type', 'gt_text'])
    #     df['gt_text_len'] = df['gt_text'].apply(len)
    #     dfs.append(df)

    # all_dfs[website_path.parts[-1].split(".pickle")[0]] = dfs

    # # assert df['node-type'].value_counts()['PAST_CLIENT'] > 0, "There is a page that doesn't contain any Past Client"
    # # break
website_data['0000'][0]

# %%
import numpy as np
for enum, (website, dfs) in enumerate(all_dfs.items()):
    # print(website)
    for df in dfs:
        if np.any(df["gt_text_len"] > 1):
            display(df)
            break

# %%
node_count = {'none':[], 'PAST_CLIENT':[], 'nonempty-none':[]}
text_length = {'none':[], 'PAST_CLIENT':[]}

websites = []
pages = []
positive_dfs = []
negative_dfs = []
all_df = pd.DataFrame()

websites_iterator = tqdm(websites_data_path)
for website_path in websites_iterator:
    websites_iterator.set_description(f"Processing: {website_path}")
    website_data = pd.read_pickle(website_path)
    no_past_client_pages = []
    
    for page_index in website_data.keys():
        website = str(website_path.parts[-1]).split('.pickle')[0]
        websites.append(website)
        pages.append(page_index)
        
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'gt_field', 'gt_text', 'node_attribute'])
        if len(df) > 0:
            df["website"] = website
            all_df = all_df.append(df)

    #         node_distribution = df['gt_field'].value_counts()

    #         if 'PAST_CLIENT' not in node_distribution:
    #             node_distribution['PAST_CLIENT'] = 0
    #             no_past_client_pages.append(page_index)
    #             negative_dfs.append(df)
    #         else:
    #             positive_dfs.append(df)

    #         node_ratio = node_distribution.get('PAST_CLIENT', 0) / (1 + node_distribution.get('none', 0))
    #         if node_ratio > 0.6:
    #             print(f"Strange - ratio! {node_ratio:.2f} | {website} | {page_index}")
    #             print(node_distribution)
    #             print()
                
    #         past_clients = node_distribution.get('PAST_CLIENT', 0)
    #         if past_clients > 100:
    #             print(f"Strange - many PAST CLIENTS! {past_clients} | {website} | {page_index}")
    #             print()

    #         df = df[df['text'] != '']
    #         df['text_len'] = df['text'].apply(len)
            
    #         non_empty_node_count = df['gt_field'].value_counts().get('none', 0)
            
    #         node_count['none'].append(node_distribution.get('none', 0))        
    #         node_count['PAST_CLIENT'].append(node_distribution.get('PAST_CLIENT', 0))
    #         node_count['nonempty-none'].append(non_empty_node_count)

            
    #         text_avg = pd.DataFrame(df.groupby('gt_field').mean('text_len'))['text_len']        
                            
    #         text_length['none'].append(text_avg['none'])

    #         if 'PAST_CLIENT' in text_avg:
    #             text_length['PAST_CLIENT'].append(text_avg['PAST_CLIENT'])

    # # print(f"{website} - No past clients: {len(no_past_client_pages)} out of {len(website_data.keys())}")

# %%
print(len(websites_data_path))

def read_data(website_path):
    dfs = pd.DataFrame()
    website_data = pd.read_pickle(website_path)
    for page_index in website_data.keys():
        website = str(website_path.parts[-1]).split('.pickle')[0]
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'gt_field', 'gt_text', 'node_attribute', 'node_tag'])
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
all_dfs['gt_text_len'] = all_dfs['gt_text'].apply(len) 

# %% [markdown]
# # Label Analysis
# ## _node_tag_ and _node_attribute_
# By identified where it is not likely a positive label to appear we should remove those cases and limit the scope of the data in order to:
# - Reduce time in all stages 
# - Remove noisy data

# %%
print(all_dfs.columns.values)


# %%
def node_analysis(all_dfs, col:str):
    all_data_size = len(all_dfs)
    positives = all_dfs[all_dfs["gt_text_len"] > 0]
    negatives = all_dfs[all_dfs["gt_text_len"] == 0]

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
pd.DataFrame(all_dfs[all_dfs['gt_text_len']>0]['text_len'].describe()).style.format(na_rep='MISS', precision=1)  

# %%
print("Negative node text length distribution:")
pd.DataFrame(all_dfs[all_dfs['gt_text_len']==0]['text_len'].describe()).style.format(na_rep='MISS', precision=1)  

# %% [markdown]
# ## Check how long the node text can be in order to remove high memorydata
#

# %%
max_len = 10000
all_gt = all_dfs["gt_text_len"].sum()
gt_numb = all_dfs[all_dfs["text_len"] > max_len]["gt_text_len"].sum()
data_numb = len(all_dfs[all_dfs["text_len"] > max_len]) 
all_data = len(all_dfs)

text_len = all_dfs[all_dfs["text_len"] > max_len]["text_len"].sum()
all_text_len = all_dfs["text_len"].sum()

print(f"gt_numb: {gt_numb}")
print(f"data_numb: {data_numb}")
print(f"ratio:\n gt_numb_ratio: {100*gt_numb/all_gt:.2f} %\n data_numb_ratio: {100*data_numb/all_data:.2f} %\n text_len_ratio: {100*text_len/all_text_len:.2f} %")

# %%
# all_dfs

# %%
print(f" Empty text nodes: {len(all_dfs[all_dfs['text_len'] <= 1])}")

# %%
# all_dfs['text_len'] = all_dfs['text'].apply(len)
# all_dfs.groupby("node_attribute")['text_len'].describe().sort_values("count", ascending=False)

# %%
# import collections
# counts = collections.defaultdict()
# none_counts = []
# past_client_counts = []
# attributes = all_dfs['node_attribute'].value_counts().index
# for x in attributes:
#     counts = all_dfs[all_dfs['node_attribute'] == x]["gt_field"].value_counts()
#     past_client_counts.append(counts.get('PAST_CLIENT'))
#     none_counts.append(counts.get('none'))

# %%
# dd = pd.DataFrame([attributes, past_client_counts, none_counts]).T
# dd.columns = ["xpath", "past_clients", "none"]
# dd.sort_values("past_clients",ascending=False)["past_clients"].value_counts()

# %% [markdown]
# ---
