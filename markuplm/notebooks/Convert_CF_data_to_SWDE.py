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

# %% [markdown] tags=[]
# # Convert the CF data into the SWDE format
# 1. This is done per dataset type (train, develop, test)
# 2. Make sure to select only the positive subset (pages containin at least one PAST CLIENT annotation text)
# 3. Select the Tags (default = PAST_CLIENT)
# ---
# After that, this notebook will:
# 1. Clean a bit the data
#  - By adjusting the name of domains by removing any hyphen
#  - Making sure the name of the domain doesnt contain parenthesis
# 2. Remove annotations:
#  - Those that don't contain html
#  - Those that cannot be found in the html (using the same processing they do on the html) 
#  - Those annotations that are pointing to links (contain http)
# 3. Format and save the data into the appropriate SWDE format
# ---
# website = domain

# %% tags=[]
import pandas as pd
import pandavro as pdx
from ast import literal_eval
from pathlib import Path
from typing import List

from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.prepare_data import get_dom_tree

# %% [markdown] tags=[]
# # Load

# %% tags=[]
# dataset = 'train' #! Run this data preparation pipeline on a big machine 
dataset = 'develop'
dataset

# %% [markdown]
# ## Full Data

# %% tags=[]
data_path = f"../../../web-annotation-extractor/data/processed/{dataset}/dataset.avro"
df = pdx.read_avro(data_path)
df.annotations = df.annotations.apply(literal_eval)
len(df)

# %%
# #? Remove hiphen of domain
df["domain"] = df["domain"].apply(lambda x: x.replace('-', ''))

# %%
from microcosm.api import create_object_graph

def annotations_to_gt_value_and_gt_text(annotations_tag:pd.Series):    
    gt_values = annotations_tag.apply(lambda annotations: [x['value'] for x in annotations if x['value']])
    gt_text = annotations_tag.apply(lambda annotations: [x['text'] for x in annotations if x['text']])
    return gt_values, gt_text

# #? Create mapping to convert gt_value_taxonomy into gt_value
graph = create_object_graph("test")
taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])
def untaxonomize_gt_value(gt_value: str):
    gt_value_untax = taxonomy_to_value_mappings.get(gt_value)
    return gt_value_untax


# %%
tag = "PAST_CLIENT"

df[f'{tag}-annotations'] = df["annotations"].apply(lambda annotation: annotation.get(tag))
df[f'{tag}-annotations'] = df[f'{tag}-annotations'].fillna("").apply(list)

df[f"{tag}-gt_value"], df[f"{tag}-gt_text"] = annotations_to_gt_value_and_gt_text(df[f'{tag}-annotations'])
df[f"{tag}-gt_value_untax"] = df[f"{tag}-gt_value"].apply(lambda gt_value: [untaxonomize_gt_value(x) for x in gt_value])
df[f"{tag}-annotations-untax"] = df[f'{tag}-annotations'].apply(lambda annotations: [{"gt_text":annotation["text"], "gt_value_untax":untaxonomize_gt_value(annotation["value"])} for annotation in annotations])
df[f"{tag}-gt_text_len"] = df[f"{tag}-gt_text"].apply(len)

# %% [markdown]
# ## Positive Data (containing at least one annotation)

# %% tags=[]
df_positives = df[df[f"{tag}-gt_text_len"] > 0]
df_negatives = df[df[f"{tag}-gt_text_len"] == 0]

negative_fraction = 0.10

domains_20_or_less = df_negatives.groupby('domain')['url'].count()[df_negatives.groupby('domain')['url'].count() <= 20].index
domains_more_than_20 = df_negatives.groupby('domain')['url'].count()[df_negatives.groupby('domain')['url'].count() > 20].index

df_negatives_sample = df_negatives[df_negatives['domain'].isin(domains_more_than_20)].groupby("domain").sample(frac=negative_fraction, random_state=66)
df_negatives_sample = df_negatives_sample.append(df_negatives[df_negatives['domain'].isin(domains_20_or_less)])

# %%
df_positives_initial_len = len(df_positives)
print(f"Negatives: {len(df_negatives)} | Negatives sample: {len(df_negatives_sample)} | Positives:{len(df_positives)}")

# %% [markdown]
# # Clean Data

# %% [markdown]
# ## Assert that domain don't contain parenthesis
#

# %% tags=[]
assert len(df_negatives_sample[df_negatives_sample['domain'].apply(lambda x: '(' in x or ')' in x)]) == 0
assert len(df_positives[df_positives['domain'].apply(lambda x: '(' in x or ')' in x)]) == 0
# Remove parenthesis from domain in case there is (assertion above fails)
# ... df = df[~df['domain'].apply(lambda x: '(' in x or ')' in x)]

# %% [markdown]
# # Remove pages that don't have html

# %% tags=[]
pages_without_html = df_positives[df_positives['html'] == 'PLACEHOLDER_HTML']
annotations_without_html = pages_without_html[f"{tag}-gt_text_len"].sum()
print(f"Pages removed: {len(pages_without_html)}")
print(f"Annotations removed: {annotations_without_html}")
df_positives = df_positives[df_positives['html'] != 'PLACEHOLDER_HTML']

# %%
pages_without_html = df_negatives_sample[df_negatives_sample['html'] == 'PLACEHOLDER_HTML']
print(f"Pages removed: {len(pages_without_html)}")
df_negatives_sample = df_negatives_sample[df_negatives_sample['html'] != 'PLACEHOLDER_HTML']

# %% [markdown]
# ## Remove pages that are not strictly HTML

# %%
import lxml
# TODO: Deal with XLM cases
def get_only_html(t):
    text = 'NOT HTML'
    try:
        text = lxml.html.fromstring(t)
        return t
    except:
        return text


# %%
print(len(df_positives))
all_pages = df_positives['html'].apply(lambda x: get_only_html(x))
positive_non_html_pages = df_positives[all_pages == 'NOT HTML']
df_positives = df_positives[all_pages != 'NOT HTML']
print(len(df_positives))

# %%
print(len(df_negatives_sample))
all_pages = df_negatives_sample['html'].apply(lambda x: get_only_html(x))
negative_non_html_pages = df_negatives_sample[all_pages == 'NOT HTML']
df_negatives_sample = df_negatives_sample[all_pages != 'NOT HTML']
print(len(df_negatives_sample))

# %%
non_html_pages = positive_non_html_pages.append(negative_non_html_pages)
print(f"non_html_pages: {len(non_html_pages)}")
if len(non_html_pages) > 0:
    save_path = f"{dataset}-non_html_pages({len(non_html_pages)}).csv"
    print(f"Save path: {save_path}")
    non_html_pages.to_csv(save_path)

# %% [markdown]
# # Remove annotations that cannot be found in the xpaths of the html

# %%
initial_amount_of_label = df_positives[f"{tag}-gt_text_len"].sum()
print(f"Initial amount of labels: {initial_amount_of_label}")

all_new_annotations = []

for i, row in tqdm(df_positives.iterrows()):
    if not row.isnull()[f'{tag}-gt_text']:
        clean_dom_tree = get_dom_tree(row['html'], 'website')
        
        annotations_that_can_be_found = []
        annotations_that_cannot_be_found = []
        for text_annotation in row[f'{tag}-gt_text']:
            for node in clean_dom_tree.iter():
                if node.text:
                    if text_annotation in node.text:
                        annotations_that_can_be_found.append(text_annotation)
                        break
                if node.tail:
                    if text_annotation in node.tail:
                        annotations_that_can_be_found.append(text_annotation)
                        break
                # for html_tag, xpath_content in node.items():
                #     if text_annotation in xpath_content:
                #         annotations_that_can_be_found.append(text_annotation)
                #         break
            annotations_that_cannot_be_found.append(text_annotation)
            
        if len(annotations_that_cannot_be_found) > 0:
            print(f"Cannot be found in {i} \t: {annotations_that_cannot_be_found}")
            print()

        all_new_annotations.append(annotations_that_can_be_found)
    else:
        all_new_annotations.append(None)

df_positives[f'{tag}-gt_text'] = all_new_annotations
df_positives[f"{tag}-gt_text_len"] = df_positives[f"{tag}-gt_text"].apply(len)
final_amount_of_label = df_positives[f"{tag}-gt_text_len"].sum()

# %% tags=[]
print(f"Final amount of labels: {final_amount_of_label}")
print(f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}")

# %% tags=[]
current_annotations = df_positives[f"{tag}-gt_text_len"].sum()
print(f"Annotations: {current_annotations} | Pages: {len(df_positives)}")

# %% tags=[]
print(f"With text and tail the page coverage is: {100*len(df_positives)/df_positives_initial_len:.2f} %")

# %% tags=[]
# print(f"With text, tail and xpath_content the annotation coverage is: {100*11500/12085:.2f} %") # outdated

# %% [markdown]
# ## Debug (find why annotation could not be found in html):

# %% tags=[]
# row = df.iloc[3143]
# print(row)
# clean_dom_tree = get_dom_tree(row['html'], 'website')
# print()
# for node in clean_dom_tree.iter():
#     for html_tag, xpath_content in node.items():
#         print('text:', node.text)
#         print('tail:', node.tail)
#         print('xpath:', xpath_content)        

# %%
# [x for x in df_positives_negatives['domain'].value_counts().sort_values().index if 'group' in x]

# %% [markdown]
# # Remove image link annotations

# %% tags=[]
current_annotations = df_positives[f"{tag}-gt_text_len"].sum()
print(f"Annotations: {current_annotations} | Pages: {len(df_positives)}")

# %% tags=[]
df_positives[f"{tag}-gt_text"] = df_positives[f"{tag}-gt_text"].dropna().apply(lambda annotations: [annotation  for annotation in annotations if 'http' not in annotation])

# %% tags=[]
df_positives[f"{tag}-gt_text_len"] = df_positives[f"{tag}-gt_text"].apply(len)
current_annotations = df_positives[f"{tag}-gt_text_len"].sum()
print(f"Annotations: {current_annotations} | Pages: {len(df_positives)}")

# %% [markdown]
# # Remove samples without annotation

# %% tags=[]
df_positives = df_positives[df_positives[f"{tag}-gt_text_len"] > 0]

# %% tags=[]
current_annotations = df_positives[f"{tag}-gt_text_len"].sum()
print(f"Annotations: {current_annotations} | Pages: {len(df_positives)}")

# %% [markdown]
# # Remove domains in negatives that are not in the positives

# %%
df_negatives_sample = df_negatives_sample[df_negatives_sample['domain'].isin(df_positives['domain'])]

# %% [markdown]
# # Checks

# %% [markdown]
# ## The number of domains in negatives are the same as in positives

# %%
print(f"Positive Domains: {len(set(df_positives['domain']))} | Negative Domains: {len(set(df_negatives_sample['domain']))}")
assert len(set(df_negatives_sample['domain']) - set(df_positives['domain'])) == 0, 'Negatives have a domain that positive doesnt have!'

# %% [markdown]
# ## The percentage of negative data is still the same (not more than 10%relative difference)

# %%
df_negatives_positive_domain = df_negatives[df_negatives['domain'].isin(df_positives['domain'])]
final_negative_fraction = len(df_negatives_sample) / len(df_negatives_positive_domain)
print(f" {len(df_negatives_sample)} | {len(df_negatives_positive_domain)} | {100*final_negative_fraction:.4f} %")
assert negative_fraction - 0.01 < final_negative_fraction < negative_fraction + 0.01

# %% [markdown]
# # Add negatives back

# %% tags=[]
df_positives_negatives = df_positives.append(df_negatives_sample)
print(f"Positive negatives: '{len(df_positives_negatives)}' pages and '{len(set(df_positives_negatives['domain']))}' domains")

# %% [markdown]
# # Save intermediate Data

# %% tags=[]
data_path

# %% tags=[]
save_path = data_path.replace('.avro', f'_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl')
print(f"Saving file: {save_path}")
df_positives_negatives.to_pickle(save_path)

save_path = data_path.replace(".avro", f"_pos({len(df_positives)})_intermediate.pkl")
print(f"Saving file: {save_path}")
df_positives.to_pickle(save_path)

# %% [markdown] tags=[]
# # Format and Save data

# %% tags=[]
import os
import shutil

pageid_url_mapping = {}

raw_data_folder = Path.cwd().parents[2] / f'swde/my_data/{dataset}/my_CF_sourceCode'

if os.path.exists(raw_data_folder): #! Uncomment
    print(f'Are you sure you want to remove this folder? (y/n) \n{raw_data_folder}')
    # answer = input()
    answer = 'y'
    if answer == 'y':
        try:
            shutil.rmtree(raw_data_folder)
            print(f"REMOVED: {raw_data_folder}")
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

# else:
#     print(f"File '{groundtruth_data_path}' not found in the directory")
    
domains = list(df_positives_negatives.domain.value_counts().index)

groundtruth_data_path = raw_data_folder / 'groundtruth'
groundtruth_data_path.mkdir(parents=True, exist_ok=True) #! Uncomment

for e, domain in enumerate(domains):        
    df_domain = df_positives_negatives[df_positives_negatives.domain == domain]
    
    print(f"{e:>3}: {len(df_domain):>5} page(s) - {domain:>25}")
    
    domain_annotations = {}
    
    page_count = 0
    domain_len = len(df_domain)
    
    for enum, df_page in df_domain.iterrows():
        html = df_page['html']
        raw_data_path = raw_data_folder / 'WAE' / f"{domain}({domain_len})"
        raw_data_path.mkdir(parents=True, exist_ok=True)
        raw_data_path = (raw_data_path / str(page_count).zfill(4)).with_suffix('.htm')

        name = str(page_count).zfill(4) + '.htm'
        pageid = f"{domain}.pickle-{name}"
        url = df_page['url']
        pageid_url_mapping[pageid] = [url]
        
        Html_file = open(raw_data_path, "w") #! Uncomment
        Html_file.write(html)
        Html_file.close()
        
        page_count += 1

        # #? Get groundtruth for page for each tag
        for tag in ["PAST_CLIENT"]:            
            domain_annotations[tag] = domain_annotations.get(tag, [])
            if not df_page.isnull()[f'{tag}-gt_text']:
                annotations = df_page[f'{tag}-gt_text']                
                # #? Remove image links from text annotation
                annotate = [annotation.strip() if (annotation and 'http' not in annotation.strip()) else '' for annotation in annotations]
            else:
                annotate = []
            # print(f'annotate: \n{annotate} - {len(annotate)}')            
            domain_annotations[tag].append(annotate)            
        # print()
        # if raw_data_path.name == '0042.htm':
        #     break
        
    # #? Save groundtruth    
    for tag, page_annotations in domain_annotations.items():
        groundtruth_data_tag_path = groundtruth_data_path / f"{domain}-{tag}.txt"
        # groundtruth_data_tag_path = groundtruth_data_path / f"{domain}-{tag}.csv" #! Uncomment once I change to CSV
        print(groundtruth_data_tag_path)

        page_annotations_df = pd.DataFrame(page_annotations)
        
        # #? Count number of annotations
        page_annotations_df['number of values'] = page_annotations_df.T.count()        
        
        # #? Invert columns order 
        cols = page_annotations_df.columns.tolist()
        page_annotations_df = page_annotations_df[cols[::-1]] 
        
        # #? Get page index
        page_annotations_df.reset_index(inplace=True)
        page_annotations_df['index'] = page_annotations_df['index'].apply(lambda x: str(x).zfill(4))
        
        # #? Add one extra row on the top
        page_annotations_df.loc[-1] = page_annotations_df.count()  # adding a row
        page_annotations_df.index = page_annotations_df.index + 1  # shifting index
        page_annotations_df = page_annotations_df.sort_index()
        
        page_annotations_df.to_csv(groundtruth_data_tag_path, sep="\t", index=False) #! Uncomment

# %%
pd.to_pickle(pageid_url_mapping, f"/data/GIT/swde/my_data/{dataset}/my_CF_sourceCode/pageid_url_mapping.pkl")

# %% [markdown]
# # Final Stats

# %% [markdown] tags=[]
# ## Number of Domains

# %% tags=[]
df_positives_negatives.domain.value_counts()

# %% [markdown]
# ## Number of Pages

# %%
df

# %%
current_annotations = len([y for x in df_positives_negatives.dropna(subset=[f'text-{tag}'])[f'text-{tag}'].values for y in x])
print(f"Annotations: {current_annotations} | Pages: {len(df_positives_negatives)}")

# %% [markdown]
# ---
