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
#     display_name: Python 3.8.0 ('html39')
#     language: python
#     name: python3
# ---

# %%
from new_segmenter import Segmenter
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.evaluations.metric_functions import get_reconciliations_metrics_for_all_domains, calculate_metrics_for_dataset
import multiprocessing as mp
# from web_annotation_extractor.common.utils.general_utils import deserialize_annotations

from tqdm import tqdm
import pandavro as pdx
from pathlib import Path
from ast import literal_eval
import pandas as pd
from marquez.enums.annotation import EntityTag

# %%
# %load_ext autoreload
# %autoreload 2

# %%
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 4)
pd.set_option("display.min_rows", 4)

# %% [markdown]
# # Taxonomy Known companies duplicated Names

# %%
graph = create_object_graph("gazetteers")

known_company_taxonomy = []
for company in graph.known_company_taxonomy:
    if company.is_demo_company is False and company.deprecated is False:
        known_company_taxonomy.append(company)

company_name_to_uri_map = dict({
    (company.name, company.uri)
    for company in known_company_taxonomy
})
uri_to_company_name_map = dict({
    (company.uri, company.name)
    for company in known_company_taxonomy
})


# %% [markdown]
# # Load Data

# %%
def load_data(datasets):
    all_df = pd.DataFrame()
    for dataset in datasets: #! Only Train for now
        # data_path = f"/data/GIT/web-annotation-extractor/data/processed/{dataset}/dataset.avro"
        df = pdx.read_avro(data_path)    
        print(f"{dataset}: {len(df)}")
        all_df = pd.concat([all_df, df])
    print(len(all_df))
    return all_df

def get_positives(df, tag):
    df["annotations"] = df["annotations"].apply(deserialize_annotations)    
    df[tag.name] = df['annotations'].apply(lambda row: row.get(tag))
    return df.dropna(subset=tag.name)

def get_positives(df, tag_name):
    df["annotations"] = df["annotations"].apply(literal_eval)
    df[tag_name] = df['annotations'].apply(lambda row: row.get(tag_name))
    return df.dropna(subset=tag_name)


# %%
tag = EntityTag.PAST_CLIENT

overwrite = False

# datasets = ['train', 'develop', 'test']
datasets = ['train']
# datasets = ['develop']

if len(datasets) == 3: 
    dataset_name = 'all'
else:
    dataset_name = "_".join(datasets)

# %%
# data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1765)_neg(4086)_intermediate.pkl"
# classified_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred_with_img.pkl"
predicted = False
if datasets[0] == 'train':
    predicted_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/train_df_pred_after_training(522031).pkl"
    # not_predicted_nodes_data_path = "/data/GIT/node_classifier_with_imgs/train/processed_dedup.pkl"
    not_predicted_nodes_data_path = "/data/GIT/prepared_data/node_classifier_with_imgs/train/processed_dedup.pkl"
    data_path = "/data/GIT/web-annotation-extractor/data/processed/train/dataset_pos(4319)_neg(13732)_intermediate.pkl"

if datasets[0] == 'develop':
    predicted_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred_after_training(178346).pkl"
    # not_predicted_nodes_data_path = "/data/GIT/node_classifier_with_imgs/develop/processed_dedup.pkl"
    not_predicted_nodes_data_path = "/data/GIT/prepared_data/node_classifier_with_imgs/develop/processed_dedup.pkl"
    data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1830)_neg(4587)_intermediate.pkl"

df = pd.read_pickle(data_path)
df = df.rename(columns={"PAST_CLIENT-annotations": "PAST_CLIENT"})
print(len(df))
if predicted:
    predicted_df = pd.read_pickle(predicted_nodes_data_path)
else:
    predicted_df = pd.read_pickle(not_predicted_nodes_data_path)
print(len(predicted_df))

# %%
if predicted:
    classified_df = predicted_df.copy()
    threshold = 0.0
    classified_df = classified_df[classified_df["node_prob"] > threshold]
    len(classified_df)
    
else:
    classified_df = predicted_df.explode("nodes").reset_index()
    # # ? Join expanded nodes into df
    classified_df = classified_df.join(
        pd.DataFrame(
            classified_df.pop("nodes").tolist(),
            columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
        )
    )
print(len(classified_df))

# %% [markdown]
# # Load Model

# %%
s = Segmenter()
print(s.number_of_companies(), s.number_of_regexes())
# s.number_of_companies()
# s.number_of_regexes()

# %%
s.augment_company_names_with_training_data(df)
print(s.number_of_companies(), s.number_of_regexes())

s.transform_regexes() 
print(s.number_of_companies(), s.number_of_regexes())

s.remove_duplicated_regexes_and_sort()
print(s.number_of_companies(), s.number_of_regexes())

# %% [markdown]
# # Get the Positives only

# %%
# #! Transform the annotations that were supposed to be images to images
a = classified_df[classified_df["node_gt_tag"] != 'none']
a = a[a.apply(lambda row : 'http' == row["node_text"][:4], axis=1)].explode('node_gt_text')
a['node_text'] = a['node_text'].apply(lambda x: x.lower())
a['node_gt_text'] = a['node_gt_text'].apply(lambda x: x.lower())

a_img = a[a.apply(lambda x: 'http' in x['node_gt_text'],axis=1)][["url", "domain", "xpath",  "node_text", "node_gt_text"]]
a_text = a[a.apply(lambda x: 'http' not in x['node_gt_text'],axis=1)][["url", "domain", "xpath",  "node_text", "node_gt_text"]]

print("Total:", len(a))
print("Annotations that are images:", len(a_img))
print("Annotations that are text:", len(a_text))

mapping_text_to_img = a_text.set_index(['url', 'xpath', 'node_gt_text']).to_dict()['node_text']

classified_df["node_gt_text"] = classified_df.apply(lambda row: [mapping_text_to_img.get((row['url'], row['xpath'], x.lower())) if mapping_text_to_img.get((row['url'], row['xpath'], x.lower())) else x for x in row["node_gt_text"] ],axis=1)

# %%
a = classified_df[classified_df["node_gt_tag"] != 'none']
a = a[a.apply(lambda row : 'http' == row["node_text"][:4], axis=1)].explode('node_gt_text')
a['node_text'] = a['node_text'].apply(lambda x: x.lower())
a['node_gt_text'] = a['node_gt_text'].apply(lambda x: x.lower())

a_img = a[a.apply(lambda x: 'http' in x['node_gt_text'],axis=1)][["url", "domain", "xpath",  "node_text", "node_gt_text"]]
a_text = a[a.apply(lambda x: 'http' not in x['node_gt_text'],axis=1)][["url", "domain", "xpath",  "node_text", "node_gt_text"]]

print("Total:", len(a))
print("Annotations that are images:", len(a_img))
print("Annotations that are text:", len(a_text))

assert len(a_text) == 0

# %%
# # #? Get positives that don't contain images:
positives_with_no_images_indices = list(classified_df[classified_df.apply(lambda row: 'http' == row["node_text"][:4], axis=1)].index & classified_df[classified_df["node_gt_tag"] != 'none'].index)
classified_df = classified_df.loc[positives_with_no_images_indices]

# # #? Get positives that contain images:
classified_df = classified_df[classified_df["node_gt_tag"] != 'none']

len(classified_df)

# %%
# #! Filter out some specific domains for debug reasons
# # domain_names = ["hawthorneadvertising.com", "workingfamilies.org.uk", "themplc.co.uk", "elliscasting.com", "gobeyondpartners.com", "hybridcollective.tv"]
# domain_names = ['wc.com']
# # classified_df = classified_df[~classified_df['domain'].isin(domain_names)]
# classified_df = classified_df[classified_df['domain'].isin(domain_names)]

# # df = df[~df['domain'].isin(domain_names)]
# df = df[df['domain'].isin(domain_names)]

# %% [markdown]
# # Transform

# %%
# # #? Transform all nodes
p = mp.Pool(mp.cpu_count())
transformed_texts = []
for transformed_text in p.imap(s.transform_texts, classified_df["node_text"], chunksize = 50):
    transformed_texts.append(transformed_text)
print(len(transformed_texts))
classified_df["node_text_t"] = transformed_texts

# %% [markdown]
# # Match

# %%
p = mp.Pool(mp.cpu_count())
matches = []
for match in p.imap(s.find_companies, classified_df["node_text_t"], chunksize = 50):
    matches.append(match)
print(len(matches))
classified_df["gaz_matches"] = matches

# %% [markdown]
# ## Predictions

# %%
predited_df = classified_df.copy(deep=True)

# predited_df["matches"] = predited_df["rel_matches"]
predited_df["matches"] = predited_df["gaz_matches"]
# predited_df["matches"] = predited_df["both_matches"]

predited_df = predited_df[["url", "matches"]]
predited_df = predited_df.groupby("url").agg({'matches': 'sum'})

# %%
merge = df.set_index('url').join(predited_df).reset_index()
merge["matches"] = merge["matches"].fillna('').apply(list)

# %%
merge['predicted_tag'] = merge['matches']

# %%
# # #? Get the gt companies
merge["gt_tag_with_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value')])
merge["gt_tag_without_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# %% [markdown]
# # Evaluate

# %% [markdown]
# ### With Tag Images

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %% [markdown]
# ### Without Tag Images

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
classified_df[classified_df["domain"] == "wc.com"][["gaz_matches", "PAST_CLIENT-gt_value_untax"]]
# get_reconciliations_metrics_for_all_domains(merge[merge["domain"] == "hybridcollective.tv"], gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
# merge[merge["domain"] == "hybridcollective.tv"][["matches", "PAST_CLIENT-gt_value_untax"]]

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',12, 'display.min_rows',12)
domain_metrics.sort_values("FN", ascending=False)

# %% [markdown]
# # Error Analysis

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

# #! I want to see the FN segmentations that the model missed, and what are the regexes of these companies that the model couldn't find
domain_metrics = domain_metrics.reset_index()
domain_metrics = domain_metrics.rename({'index':"domain"},axis=1)
fn_error = domain_metrics[domain_metrics["FN"] > 0].sort_values("FN",ascending=False).explode("FN_seg").sort_values("FN_seg")
mapping_compId_regexes = pd.DataFrame(s.companies_library).set_index("company_id")["regexes"].to_dict()
fn_error["regexes"] = fn_error["FN_seg"].apply(lambda row: mapping_compId_regexes.get(row[0]))
fn_error

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
#? ["hawthorneadvertising.com", "elliscasting.com", "gobeyondpartners.com", ""]: Showed me that the annotators annotated image as text
#? ["workingfamilies.org.uk"]: Showed me that the annotations are not in a normal tag
#? ["themplc.co.uk"]: Showed me that the model was skipping annotations when there was BR as html tag
#? ["hybridcollective.tv"]: Showed me that the model predicted the companies, but still were marked as FN
#? ["wc.com"]: Showed me that they are as negatives, eventhough the nodes contain the positives - Exploring...

domain_names = ["hawthorneadvertising.com", "workingfamilies.org.uk", "themplc.co.uk", "elliscasting.com", "gobeyondpartners.com", "hybridcollective.tv", "wc.com"]
domain_name = domain_names[-1]
print(domain_name)

# %% [markdown]
# #### Check what are the companies that it should appear in the node_dataset

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
fn_error[fn_error['domain'] == domain_name]["FN_seg"]

# %% [markdown]
# #### Check if the FN node_gt_text are in the node_text

# %%
gt_text = list(fn_error[fn_error['domain'] == domain_name]["FN_seg"].apply(lambda x: x[1]).values)
classified_df[classified_df['domain'] == domain_name][classified_df[classified_df['domain'] == domain_name]["node_text"].apply(lambda row: any([True if x in row else False for x in gt_text ]))][["url", "node_text", "node_gt_text", "gaz_matches"]]

# %% [markdown]
# ### Check if the FN node_gt_text are in the html

# %%
html = df[df['domain'] == domain_name].iloc[:1]["html"].values[0]
url = df[df['domain'] == domain_name].iloc[:1]["url"].values[0]
print(url)
for  x in domain_metrics[domain_metrics['domain'] == domain_name]["FN_seg"].values[0]:
    if x[1] in html:
        print('IN', x)
    else:
        print('NOT IN', x)

# %% [markdown]
# ### Check if the positive classified_df has nodes of this domain

# %%
classified_df[classified_df['domain'] == domain_name]

# %%
# #! Check if node_dataset has the same annotations as page_dataset per domain
# #! Get the pages which has gt_value (which were probably dropped when converting html into nodes)
total_number_gt_value_in_page_dataset = 0
total_number_gt_node_in_node_dataset = 0
total_number_gt_value_diff = 0
pages_that_contain_gt_value_in_page_but_not_in_node_dataset = []
for enum, domain in enumerate(sorted(list((set(fn_error['domain']))))):
    domain_FN_company_ids = fn_error[fn_error['domain'] == domain]['FN_pred'].iloc[0]
    domain_nodes = classified_df[classified_df['domain'] == domain]

    domain_pages = df[df['domain'] == domain]
    fn_pages = domain_pages[domain_pages["PAST_CLIENT-gt_value"].apply(lambda row: any([True if x in domain_FN_company_ids else False for x in row]))]
    gt_value_in_page_dataset = set(domain_pages["PAST_CLIENT-gt_value"].explode().dropna())
    gt_value_in_node_dataset = set(domain_nodes["PAST_CLIENT-gt_value"].explode().dropna())
    gt_diff = gt_value_in_page_dataset - gt_value_in_node_dataset
    # print(f'Domain difference: {domain}')

    gt_value_in_page_but_not_in_node_dataset = domain_pages[domain_pages['PAST_CLIENT-gt_value'].apply(lambda row: any([True if x in gt_diff else False for x in row]))]['url'].values
    pages_that_contain_gt_value_in_page_but_not_in_node_dataset.extend(gt_value_in_page_but_not_in_node_dataset)
    total_number_gt_value_in_page_dataset += len(gt_value_in_page_dataset)
    total_number_gt_node_in_node_dataset += len(gt_value_in_node_dataset)
    total_number_gt_value_diff += len(gt_diff)
    # print()
    # display(fn_nodes[["node_gt_value", "node_gt_text", "url", "domain"]])
    # if enum==
    
print(f"total_number_gt_value_in_page_dataset: {total_number_gt_value_in_page_dataset}")
print(f"total_number_gt_node_in_node_dataset: {total_number_gt_node_in_node_dataset}")
print(f"total_number_gt_value_diff: {total_number_gt_value_diff}")

print(f"Pages in page_dataset containing those urls: {len(df[df['url'].isin(pages_that_contain_gt_value_in_page_but_not_in_node_dataset)])}")
print(f"Pages in node_dataset containing those urls: {len(classified_df[classified_df['url'].isin(pages_that_contain_gt_value_in_page_but_not_in_node_dataset)])}")

# %% [markdown]
# ### Check if the any page (positive or negative) node_text - to see if the node_gt_text appear in the node_text

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)
domain_page_with_annotations = "https://www.wc.com/Firm"
all_classified_df = predicted_df[predicted_df['domain']==domain_name].explode("nodes").reset_index()
all_classified_df = all_classified_df.join(pd.DataFrame(all_classified_df.pop("nodes").tolist(), columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"]))
all_classified_df[all_classified_df['url'] == domain_page_with_annotations][["domain", "url", "node_text", "node_gt_tag","node_gt_text"]]


# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

all_classified_df[~all_classified_df["node_gt_tag"].eq('none')]# "node_gt_text"

# %%
all_classified_df[all_classified_df[["url", "node_text"]].apply(lambda x: 'Pfizer' in x["node_text"], axis=1)][["url", "node_text"]]

# %% [markdown]
# ### If the page is in the node_dataset but the nodes don't show the node_gt_text, check why converting html to nodes didn't include these nodes

# %% [markdown]
# #### Permissive approach to create nodes from html 

# %%
# #! Try to convert these pages into nodes
from lxml import etree
import lxml
dom_tree = etree.ElementTree(lxml.html.fromstring(html))

# %%
# # #! FROM prepare_data.py

# import collections
# matched_xpaths = []  # The resulting list of xpaths to be returned.
# current_xpath_data = dict()  # The resulting dictionary to save all page data.

# gt_text_in_nodes = dict()  # A list of the gt_text in each xpath node

# overall_xpath_dict = collections.defaultdict(set)

# current_page_nodes_in_order = []
# is_truth_value_list = []
# min_node_text_size = 2
# # max_node_text_size = 100_000_000
# max_node_text_size = 10_000
# for node in dom_tree.iter():
#     # The value can only be matched in the text of the node or the tail.
#     node_text_dict = {
#         "node_text": node.text,
#         "node_tail_text": node.tail,
#     }  # ?The only nodes that are considered here are the node.text and node.tail

#     for text_part_flag, node_text in node_text_dict.items():
#         if node_text:
#             if (
#                 node.tag != "script"
#                 and "javascript" not in node.attrib.get("type", "")
#                 and min_node_text_size <= len(node_text.strip()) < max_node_text_size
#             ):  #! Remove java/script and min_node_text # TODO (Aimore): Make this comparisons more explicity and descriptive
#                 # """Matches the ground truth value with a specific node in the domtree.

#                 node_attribute = node.attrib.get("type", "")
#                 node_tag = node.tag
#                 node_text_split = node_text.split("--BRRB--")
#                 len_brs = len(node_text_split)  # The number of the <br>s.
#                 for index, etext in enumerate(node_text_split):

#                     if text_part_flag == "node_text":
#                         xpath = dom_tree.getpath(node)

#                     elif text_part_flag == "node_tail_text":
#                         xpath = dom_tree.getpath(node) + "/tail"

#                     if len_brs >= 2:
#                         xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]

#                     # clean_etext = clean_spaces(etext)
#                     clean_etext = etext

#                     # ? Update the dictionary.
#                     current_xpath_data[xpath] = clean_etext
#                     overall_xpath_dict[xpath].add(clean_etext)
#                     current_page_nodes_in_order.append(
#                         (clean_etext, xpath, node_attribute, node_tag)
#                     )

#                     # ? Clean the groundtruth and the node text. Check if the groundtruth is in the node text.
#                     # clean_etext = clean_format_str(clean_etext)

#                     # ? Create node ground truth by checking if the the gt_text is in the clean node_text
#                     # gt_text_in_node = []
#                     # for gt_value in clean_gt_values:
#                     #     if f" {gt_value.strip()} ".lower() in f" {clean_etext.strip()} ".lower():
#                     #         gt_text_in_node.append(gt_value)
#                     #         matched_xpaths.append(xpath)
#                     #         is_truth_value_list.append(
#                     #             len(current_page_nodes_in_order) - 1
#                     #         )
#                     #         # break #! I am not sure why Iadded this break, I'm commenting it because I think all gt_values should be added in a node

#                     # if len(matched_xpaths) == 0:
#                     #     gt_text_in_nodes[xpath] = []
#                     # else:
#                     #     gt_text_in_nodes[xpath] = gt_text_in_node

# %% [markdown]
# #### Original approach to create nodes from html 

# %%
from markuplmft.fine_tuning.run_swde.featurizer import get_dom_tree, Featurizer

featurizer = Featurizer()
featurizer.get_nodes()

# %%
pd.DataFrame(featurizer.get_nodes(html))

# %%
dom_tree = get_dom_tree(html)

page_nodes = []
min_node_text_size, max_node_text_size = 2, 10_000

for node in dom_tree.iter():
    node_text_dict = {
        "node_text": node.text,
        "node_tail_text": node.tail,
        "node_attrib.src": node.attrib.get('src'),
        "node_attrib.alt": node.attrib.get('alt'),
    }
    for text_part_flag, node_text in node_text_dict.items():
        if node_text:
            if (
                node.tag != "script"
                and "javascript" not in node.attrib.get("type", "")                        
                and min_node_text_size <= len(node_text.strip()) < 10*max_node_text_size
            ):  #! Remove java/script and min_node_text # TODO (Aimore): Make this comparisons more explicity and descriptive

                # node_attribute = node.attrib.get("type", "")
                # node_tag = node.tag
                if text_part_flag in ["node_text", "node_tail_text"]:
                    node_text_split = node_text.split("--BRRB--")
                    len_brs = len(node_text_split)  # The number of the <br>s.

                    for index, etext in enumerate(node_text_split):
                        if min_node_text_size <= len(etext.strip()) < max_node_text_size:
                            if text_part_flag == "node_text":
                                xpath = dom_tree.getpath(node)

                            elif text_part_flag == "node_tail_text":
                                xpath = dom_tree.getpath(node) + "/tail"

                            if len_brs >= 2:
                                xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]

                            page_nodes.append((xpath, etext, "none", []))
                else:
                    xpath = dom_tree.getpath(node)
                    page_nodes.append((xpath, node_text, "none", []))

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',500, 'display.min_rows',500)
pd.DataFrame(page_nodes)

# %% [markdown]
# ### Analyse how much does the annotators got wrong the gt_text text mixed with image

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
classified_df[classified_df['node_gt_text'].apply(len) > 1][['node_gt_text', 'node_text']]

# %% [markdown]
# ## FN Error Analysis

# %%
merge["gt_tag_without_img_len"] = merge["gt_tag_without_img"].apply(len)
merge.sort_values("gt_tag_without_img_len")

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
merge[merge["gt_tag_without_img_len"] > 0]["gt_tag_without_img"].loc[4305:4305]
# .apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# for x in merge['gt_tag_without_img'].loc[4305:4305].values[0]:
#     print(x)
#     print()

# for x in merge['PAST_CLIENT'].loc[4305:4305].values[0]:
#     print(x['text'], x['value'])
#     print()

# %%
# #? Create a mapping from gt_text to gt_value to be used on the nodes
classified_df['mapping'] = classified_df.apply(lambda row: {x['text']:x['value'] for x in row["PAST_CLIENT-annotations"]}, axis=1) 
# #? Apply this mapping and get gt_value per node
classified_df['node_gt_value'] = classified_df.apply(lambda row: [row['mapping'].get(x) for x in row["node_gt_text"] if row['mapping'].get(x) and 'http' not in x], axis=1) 

# %%
classified_df["gaz_matches_len"] = classified_df["gaz_matches"].apply(len)
classified_df = classified_df.sort_values("gaz_matches_len")

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 100, 'display.max_rows',3, 'display.min_rows',3)

classified_df["gaz_matches_company_id"] = classified_df["gaz_matches"].apply(lambda row: [x['company_id'] for x in row])
classified_df["gaz_matches_regexes"] = classified_df["gaz_matches"].apply(lambda row: [x['matches'] for x in row])

# %%
classified_df["node_companyId_TP"] = classified_df.apply(lambda row: [company_id for company_id in row["gaz_matches_company_id"] if company_id  in row["node_gt_value"]], axis=1)
classified_df["node_companyId_FP"] = classified_df.apply(lambda row: [company_id for company_id in row["gaz_matches_company_id"] if company_id not in row["node_gt_value"]], axis=1)
classified_df["node_companyId_FN"] = classified_df.apply(lambda row: [node_gt_value for node_gt_value in row['node_gt_value'] if node_gt_value not in row["gaz_matches_company_id"]], axis=1)

classified_df["node_TP"] = classified_df["node_companyId_TP"].apply(len)
classified_df["node_FP"] = classified_df["node_companyId_FP"].apply(len)
classified_df["node_FN"] = classified_df["node_companyId_FN"].apply(len)

# %% [markdown]
# ## FP Error Analysis

# %%
node_companyId = ["node_companyId_TP","node_companyId_FP","node_companyId_FN"]
node_count = ["node_TP","node_FP","node_FN"]

classified_df[classified_df["node_FN"] > 0][node_companyId+node_count+["gaz_matches_regexes", "gaz_matches_regexes", "node_gt_value", "node_gt_text"]]
