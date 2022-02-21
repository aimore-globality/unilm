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

# %% tags=[] jupyter={"outputs_hidden": true}
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

# %% tags=[] jupyter={"outputs_hidden": true}
page_index = '0000'
pd.DataFrame(df[page_index], columns=['text', 'xpath', 'node-type'])

# %% [markdown] tags=[]
# ## My data 

# %% tags=[]
pd.set_option('max_colwidth', 2000)

# website_data_path = Path.cwd().parents[2] / 'swde/my_CF_processed/WAE-intralinks.com-2000.pickle'
website_data_path = Path.cwd().parents[2] / 'swde/my_CF_processed/WAE-docusign.com-9999.pickle'

website_data = pd.read_pickle(website_data_path)

# %% tags=[]
len(website_data)

# %% tags=[]
page_index = '0022'
df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
df

# %% tags=[]
df['node-type'].value_counts()

# %% [markdown]
# ## Check if all websites have at least one tag

# %% tags=[]
pd.set_option('max_colwidth', 2000)

websites_root_path = Path.cwd().parents[2] / 'swde/my_CF_processed/'
websites_data_path = list(websites_root_path.glob('[!cached]*-*9999*'))

# %% tags=[]
websites_data_path

# %% tags=[]
len(all_data)

# %% tags=[]
page_index = '0001'
df = pd.DataFrame(all_data[page_index], columns=['text', 'xpath', 'node-type'])
df

# %%
# %% tags=[]
for webiste_path in websites_data_path:
    website_data = pd.read_pickle(webiste_path)
    print(f"{webiste_path} {len(website_data)}")
    for page_index in website_data.keys():
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])
        assert df['node-type'].value_counts()['PAST_CLIENT'] > 0, "There is a page that doesn't contain any Past Client"

# %% [markdown]
# # Check how much text there are in the text of the xpaths with Past Clients

# %% tags=[]
webiste_path, page_index

# %% tags=[]
df[df['node-type'] == 'PAST_CLIENT']

# %%
for webiste_path in websites_data_path:
    website_data = pd.read_pickle(webiste_path)
    print(f"{webiste_path} {len(website_data)}")
    for page_index in website_data.keys():
        df = pd.DataFrame(website_data[page_index], columns=['text', 'xpath', 'node-type'])

# %%

# %% [markdown]
#
# %% [markdown]
# ---
