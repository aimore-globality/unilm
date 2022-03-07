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

# %% tags=[]
# vertical / website / page

# %% tags=[]
dataset = 'develop'

# %% [markdown]
# # Packed Data (Data after pack_data.py)

# %% [markdown]
# ## SWDE data 

# %% tags=[]
# data_path = '../../../swde/sourceCode/swde_small.pickle'
# data_packed = pd.read_pickle(data_path)
# len(data_packed)

# %% tags=[]
# data_packed[:1]

# %% [markdown] tags=[]
# ### Ground Truth

# %% tags=[]
# gt_path = Path('../../../swde/sourceCode/groundtruth/auto/')
# gt_file = [x for x in list(gt_path.iterdir())][0] 
# with open(gt_file) as text:
#     lines = text.readlines()
#     for l in lines:
#         print(l)

# %% [markdown]
# ## My data

# %% tags=[]
data_path = f'../../../swde/my_data/{dataset}/my_CF_sourceCode/wae.pickle'
data_packed = pd.read_pickle(data_path)
len(data_packed)

# %%
# data_packed['greatplacetowork.com']['0000']

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
# ## SWDE data 

# %% tags=[]
# website_data_path = Path.cwd().parents[2] / 'swde/swde_processed/auto-msn-3.pickle'
# df = pd.read_pickle(website_data_path)

# %% tags=[]
# page_index = '0000'
# pd.DataFrame(df[page_index], columns=['text', 'xpath', 'node-type'])

# %% [markdown] tags=[]
# ## My data 

# %% [markdown] tags=[]
# ## Check if all websites have at least one tag

# %% tags=[]
pd.set_option('max_colwidth', 2000)

websites_root_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_processed/'
print(websites_root_path)
websites_data_path = list(websites_root_path.glob('*'))
websites_data_path = [x for x in websites_data_path if 'cache' not in str(x)]
print(len(websites_data_path))
websites_data_path

# %%
assert len(websites_data_path) == len(data_packed), f"{len(websites_data_path)} != {len(data_packed)}"

# %% tags=[]
from tqdm import tqdm
for website_path in tqdm(websites_data_path):
    website_data = pd.read_pickle(website_path)
    print(f"{website_path} {len(website_data)}")
    for page_index in website_data.keys():
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
        # assert df['node-type'].value_counts()['PAST_CLIENT'] > 0, "There is a page that doesn't contain any Past Client"
    if website_path == '/data/GIT/swde/my_data/develop/my_CF_processed/direct.com.pickle':
        break

# %%
website_path

# %% tags=[]
print(website_path, len(website_data))

# %% tags=[]
df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
df

# %%
'PAST_CLIENT' in text_avg

# %%
from tqdm.notebook import tqdm

node_count = {'none':[], 'PAST_CLIENT':[], 'nonempty-none':[]}
text_length = {'none':[], 'PAST_CLIENT':[]}

websites = []
pages = []
positive_dfs = []
negative_dfs = []
for website_path in tqdm(websites_data_path):
    website_data = pd.read_pickle(website_path)
    no_past_client_pages = []
    
    for page_index in website_data.keys():
        website = str(website_path.parts[-1]).split('.pickle')[0]
        websites.append(website)
        pages.append(page_index)
        
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])

        node_distribution = df['node-type'].value_counts()

        if 'PAST_CLIENT' not in node_distribution:
            node_distribution['PAST_CLIENT'] = 0
            no_past_client_pages.append(page_index)
            negative_dfs.append(df)
        else:
            positive_dfs.append(df)

        node_ratio = node_distribution['PAST_CLIENT']/node_distribution['none']
        if node_ratio > 0.1:
            print(f"Strange - ratio! {node_ratio:.2f} | {website} | {page_index}")
            print(node_distribution)
            print()
            
        past_clients = node_distribution['PAST_CLIENT']
        if past_clients > 100:
            print(f"Strange - many PAST CLIENTS! {past_clients} | {website} | {page_index}")
            print()

        df = df[df['text'] != '']
        df['text_len'] = df['text'].apply(len)
        
        
        non_empty_node_count = df['node-type'].value_counts()['none']
        
        node_count['none'].append(node_distribution['none'])        
        node_count['PAST_CLIENT'].append(node_distribution['PAST_CLIENT'])
        node_count['nonempty-none'].append(non_empty_node_count)

        
        text_avg = pd.DataFrame(df.groupby('node-type').mean('text_len'))['text_len']        
                        
        text_length['none'].append(text_avg['none'])

        if 'PAST_CLIENT' in text_avg:
            text_length['PAST_CLIENT'].append(text_avg['PAST_CLIENT'])

    print(f"{website} - No past clients: {len(no_past_client_pages)} out of {len(website_data.keys())}")

# %%
for df in positive_dfs:
    display(df[df['node-type'] == "PAST_CLIENT"])
    print('-'*100)

# %%
df['node-type'].value_counts()

# %%
df['node-type'].value_counts()

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
