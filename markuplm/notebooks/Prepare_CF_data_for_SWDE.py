# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: markuplmft
#     language: python
#     name: markuplmft
# ---

# %% [markdown]
# # Format the CF data into the SWDE
# website = domain

# %% tags=[]
import pandas as pd
import pandavro as pdx
from ast import literal_eval
from pathlib import Path
import os   
import sys
import shutil
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.prepare_data import get_dom_tree

# %% [markdown] tags=[]
# # Load

# %% [markdown]
# ## Full Data

# %% tags=[]
# data_path = "../../../web-annotation-extractor/data/processed/develop/dataset.avro"
# df = pdx.read_avro(data_path)
# df.annotations = df.annotations.apply(literal_eval)
# len(df)

# %% [markdown]
# ## Positive Data (containing at least one annotation)

# %% tags=[]
## Create a sample that contains only positives:
# df['annotations_len'] = df['annotations'].apply(len)
# df_sample = df[df['annotations_len'] > 0]
# save_path = data_path.replace('.avro', f'_positives({len(df_sample)}).pkl')
# print(save_path)
# df_sample.to_pickle(save_path)

# %% tags=[]
data_path = "../../../web-annotation-extractor/data/processed/train/dataset_positives(11641).pkl"
df = pd.read_pickle(data_path)
len(df)

# %% [markdown]
# # Select Tags

# %% tags=[]
annotation_tags = ["PAST_CLIENT", "OFFICE_LOCATION", "CASE_STUDY"][:1]
annotation_tags

# %% [markdown]
# # Clean Data

# %% [markdown]
# ## Remove hiphen from domains

# %% tags=[]
df.domain = df.domain.apply(lambda x: x.replace('-', ''))

# %% [markdown]
# ## Assert that domain don't contain parenthesis
#

# %% tags=[]
assert len(df[df['domain'].apply(lambda x: '(' in x or ')' in x)]) == 0
# Remove parenthesis from domain in case there is (assertion above fails)
# ... df = df[~df['domain'].apply(lambda x: '(' in x or ')' in x)]

# %% [markdown]
# # Get text annotations per tag

# %% tags=[]
for tag in annotation_tags:
    print(f'- {tag}:')
    df[f'annotations-{tag}'] = df['annotations'].apply(lambda x: x.get(tag))        
    df[f'text-{tag}'] = df.dropna(subset=[f'annotations-{tag}'])[f'annotations-{tag}'].apply(lambda x: [y['text'] for y in x])        

# %% [markdown]
# # Remove samples that don't have html

# %% tags=[]
annotations_without_html = len([y for x in df[df['html'] == 'PLACEHOLDER_HTML'].dropna(subset=['annotations-PAST_CLIENT'])['text-PAST_CLIENT'] for y in x])
annotations_without_html

# %% tags=[]
df = df[df['html'] != 'PLACEHOLDER_HTML']

# %% [markdown]
# # Remove annotations that cannot be found in the xpaths of the html

# %% tags=[]
initial_amount_of_label = len([y for x in df.dropna(subset=[f'text-{tag}'])[f'text-{tag}'].values for y in x])
print(f"Initial amount of labels: {initial_amount_of_label}")

all_new_annotations = []

for i, row in tqdm(df.iterrows()):
    if not row.isnull()[f'text-{tag}']:
        clean_dom_tree = get_dom_tree(row['html'], 'website')
        
        annotations_that_can_be_found = []
        annotations_that_cannot_be_found = []
        for text_annotation in row[f'text-{tag}']:
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
        
df[f'text-{tag}'] = all_new_annotations

final_amount_of_label = len([y for x in df.dropna(subset=[f'text-{tag}'])[f'text-{tag}'].values for y in x])
print(f"Final amount of labels: {final_amount_of_label}")
print(f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}")

# %% tags=[]
len(df)

# %% tags=[]
print("With text and tail the annotation coverage is:", 9644/12085)

# %% tags=[]
print("With text, tail and xpath_content the annotation coverage is:", 11500/12085)

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

# %% tags=[]
# html = row['html']

# %% [markdown]
# # Remove image link annotations

# %% tags=[]
df[f'text-{tag}'] = df[f'text-{tag}'].dropna().apply(lambda annotations: [annotation  for annotation in annotations if 'http' not in annotation])

# %% [markdown]
# # Remove samples without annotation

# %% tags=[]
df = df[df[f'text-{tag}'].fillna('').apply(list).apply(len) > 0]
df

# %% [markdown]
# # Number of Domains

# %% tags=[]
df.domain.value_counts()

# %% [markdown] tags=[]
# # Format and Save data

# %% tags=[]
raw_data_folder = Path.cwd().parents[2] / 'swde/my_CF_sourceCode'

if os.path.exists(raw_data_folder):
    print(f'Are you sure you want to remove this folder? (y/n) \n{raw_data_folder}')
    answer = input()    
    if answer == 'y':
        try:
            shutil.rmtree(raw_data_folder)
            print(f"REMOVED: {raw_data_folder}")
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
else:
    print(f"File '{groundtruth_data_path}' not found in the directory")
    
domains = list(df.domain.value_counts().index)

groundtruth_data_path = raw_data_folder / 'groundtruth' / 'WAE'   
groundtruth_data_path.mkdir(parents=True, exist_ok=True)

for e, domain in enumerate(domains):        
    df_domain = df[df.domain == domain]
    
    print(f"{e:>3}: {len(df_domain):>5} page(s) - {domain:>25}")
    
    domain_annotations = {}    
    
    page_count = 0
    domain_len = len(df_domain)
    
    for enum, df_page in df_domain.iterrows():
        # Save html
        html = df_page['html']
        raw_data_path = raw_data_folder / 'WAE' / f"WAE-{domain}({domain_len})"
        raw_data_path.mkdir(parents=True, exist_ok=True)
        raw_data_path = (raw_data_path / str(page_count).zfill(4)).with_suffix('.htm')
        print(raw_data_path)
        
        Html_file = open(raw_data_path, "w")
        Html_file.write(html)
        Html_file.close()
        
        page_count += 1
                
        # Get groundtruth for page for each tag
        for tag in annotation_tags:            
            domain_annotations[tag] = domain_annotations.get(tag, [])
            if not df_page.isnull()[f'text-{tag}']:
                annotations = df_page[f'text-{tag}']                
                # Remove image links from text annotation
                annotate = [annotation.strip() if (annotation and 'http' not in annotation.strip()) else '' for annotation in annotations]
            else:
                annotate = []
            # print(f'annotate: \n{annotate} - {len(annotate)}')            
            domain_annotations[tag].append(annotate)            
        print()
        # if raw_data_path.name == '0042.htm':
        #     break
        
    # Save groundtruth    
    for tag, page_annotations in domain_annotations.items():
        groundtruth_data_tag_path = groundtruth_data_path / f"WAE-{domain}-{tag}.txt"
        print(groundtruth_data_tag_path)

        page_annotations_df = pd.DataFrame(page_annotations)
        
        # Count number of annotations
        page_annotations_df['number of values'] = page_annotations_df.T.count()        
        
        # Invert columns order 
        cols = page_annotations_df.columns.tolist()
        page_annotations_df = page_annotations_df[cols[::-1]] 
        
        # Get page index
        page_annotations_df.reset_index(inplace=True)
        page_annotations_df['index'] = page_annotations_df['index'].apply(lambda x: str(x).zfill(4))
        
        # Add one extra row on the top
        page_annotations_df.loc[-1] = page_annotations_df.count()  # adding a row
        page_annotations_df.index = page_annotations_df.index + 1  # shifting index
        page_annotations_df = page_annotations_df.sort_index()
        
        page_annotations_df.to_csv(groundtruth_data_tag_path, sep="\t", index=False)

# %%

# %% [markdown]
# ---
