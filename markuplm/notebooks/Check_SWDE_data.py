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

# %% tags=[]
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm

# %% tags=[]
dataset = 'develop'

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
        # website = str(website_path.parts[-1]).split('.pickle')[0]
        # websites.append(website)
        # pages.append(page_index)
        
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
all_df['node_attribute'].value_counts()

# %%
print(len(all_df))
all_df['node_attribute'].value_counts()

# %%
all_df['text_len'] = all_df['text'].apply(len)
all_df.groupby("node_attribute")['text_len'].describe()

# %%
import collections
counts = collections.defaultdict()
none_counts = []
past_client_counts = []
attributes = all_df['node_attribute'].value_counts().index
for x in attributes:
    counts = all_df[all_df['node_attribute'] == x]["gt_field"].value_counts()
    past_client_counts.append(counts.get('PAST_CLIENT'))
    none_counts.append(counts.get('none'))

# %%
dd = pd.DataFrame([attributes, past_client_counts, none_counts]).T
dd.columns = ["attribute", "past_clients", "none"]
dd

# %%
for df in positive_dfs[:10]:
    display(df[df['gt_field'] == "PAST_CLIENT"])
    print('-'*100)

# %% tags=[]
df_analysis = pd.DataFrame({'websites':websites, 'pages':pages})

df_analysis['avg_text_length-none'] = text_length['none']
df_analysis['avg_text_length-PAST_CLIENT'] = text_length['PAST_CLIENT']

df_analysis['node_count-none'] = node_count['none']
df_analysis['node_count-nonempty-none'] = node_count['nonempty-none']
df_analysis['node_count-PAST_CLIENT'] = node_count['PAST_CLIENT']

df_analysis

# %% tags=[]
pd.options.display.float_format = '{:,.1f}'.format
df_analysis.describe()

# %% tags=[]
df[df['node-type']=='none']['text'][df[df['node-type']=='none']['text'] != ''].sort_values()

# %% [markdown]
# ---
