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
#     name: markuplmft
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
dataset = 'train'

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

# %% [markdown]
# ### Ground Truth

# %% tags=[]
gt_path = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_sourceCode/groundtruth/WAE/'

df_gt = pd.DataFrame()
websites_annotations = {}

for enum, gt_file in enumerate(list(gt_path.iterdir())):
    print(f"{enum} - {gt_file}")
    df_gt = pd.read_csv(gt_file, sep='\t')
    with open(gt_file) as text:
        lines = text.readlines()
        
        # for l in lines:
        #     print(l)
        
        df_gt = df_gt.drop(['index', 'number of values'], axis=1).drop(0, axis=0).T.reset_index().drop('index',axis=1).sort_index(ascending=False)
        df_gt.columns = [str(x).zfill(4) for x in df_gt.columns]
        
        website = str(gt_file).split('groundtruth')[1].split('-')[1]
        websites_annotations[website] = df_gt
        
    display(df_gt)    

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
websites_data_path = list(websites_root_path.glob('[!cached]*-*9999*'))

# %% tags=[]
len(websites_data_path)

# %%
website_path

# %% tags=[]
for website_path in websites_data_path:
    website_data = pd.read_pickle(website_path)
    print(f"{website_path} {len(website_data)}")
    for page_index in website_data.keys():
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
        assert df['node-type'].value_counts()['PAST_CLIENT'] > 0, "There is a page that doesn't contain any Past Client"

# %% tags=[]
print(website_path, len(website_data))

# %% tags=[]
df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
df

# %% tags=[]
from tqdm import tqdm
node_count = {'none':[], 'PAST_CLIENT':[], 'nonempty-none':[]}
text_length = {'none':[], 'PAST_CLIENT':[]}

websites = []
pages = []

for website_path in tqdm(websites_data_path, f"{len(website_data):>4} = {website_path} "):
    website_data = pd.read_pickle(website_path)    
    
    for page_index in website_data.keys():
        website = str(website_path).split('WAE-')[1].split('.pickle')[0]
        websites.append(website)
        pages.append(page_index)
        
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
        assert df['node-type'].value_counts()['PAST_CLIENT'] > 0, "There is a page that doesn't contain any Past Client"

        node_distribution = df['node-type'].value_counts()        
        
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
        text_length['PAST_CLIENT'].append(text_avg['PAST_CLIENT'])                

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
