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

# %% tags=[]
import pandas as pd
from pathlib import Path

# %% tags=[]
# vertical / website / page

# %% [markdown]
# # Packed Data (Data after pack_data.py)

# %% [markdown]
# ## SWDE data 

# %% tags=[]
data_path = '../../../swde/sourceCode/swde_small.pickle'
data_packed = pd.read_pickle(data_path)
len(data_packed)

# %% tags=[]
data_packed[:1]

# %% [markdown] tags=[]
# ### Ground Truth

# %% tags=[]
gt_path = Path('../../../swde/sourceCode/groundtruth/auto/')
gt_file = [x for x in list(gt_path.iterdir())][0] 
with open(gt_file) as text:
    lines = text.readlines()
    for l in lines:
        print(l)

# %% [markdown]
# ## My data

# %% tags=[]
data_path = '../../../swde/my_CF_sourceCode/wae.pickle'
data_packed = pd.read_pickle(data_path)
len(data_packed)

# %% [markdown]
# ### Ground Truth

# %% tags=[]
gt_path = Path.cwd().parents[2] / 'swde/my_CF_sourceCode/groundtruth/WAE/'

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
page_index = '0000'
pd.DataFrame(df[page_index], columns=['text', 'xpath', 'node-type'])

# %% [markdown] tags=[]
# ## My data 

# %% tags=[]
pd.set_option('max_colwidth', 2000)

website_data_path = Path.cwd().parents[2] / 'swde/my_CF_processed/WAE-intralinks.com-2000.pickle'

all_data = pd.read_pickle(website_data_path)

# %%
len(all_data)

# %% tags=[]
page_index = '0000'
df = pd.DataFrame(all_data[page_index], columns=['text', 'xpath', 'node-type'])
df

# %%

# %% [markdown]
# ---
