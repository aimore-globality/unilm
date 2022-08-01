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
#     display_name: Python 3.9.12 ('wae39_wiki')
#     language: python
#     name: python3
# ---

# %%
from REL_NER.my_rel_segmenter import RelSegmenter

# %%
from new_segmenter import Segmenter
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.evaluations.metric_functions import get_reconciliations_metrics_for_all_domains, calculate_metrics_for_dataset

# from web_annotation_extractor.common.utils.general_utils import deserialize_annotations

from tqdm import tqdm
import pandavro as pdx
from pathlib import Path
from ast import literal_eval
import pandas as pd
from marquez.enums.annotation import EntityTag

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
# # Analyse symbols in company names

# %%
# companies_with_punc = {}
# for punc in string.punctuation:
#     for company in [x[0] for x in known_company_names.values]:
#         if punc in company:
#             company_list = companies_with_punc.get(punc, [])
#             company_list.append(company)
#             companies_with_punc[punc] = company_list

# %%
# # pd.DataFrame.from_dict()
# df = pd.DataFrame.from_dict(companies_with_punc, orient='index').T
# with pd.option_context('display.max_rows', 20): 
#     sorted_df = df.count().sort_values(ascending=False)
#     display(sorted_df)
#     display(df[sorted_df.index])

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
# datasets = ['train']
datasets = ['develop']

if len(datasets) == 3: 
    dataset_name = 'all'
else:
    dataset_name = "_".join(datasets)

# %%
# cached_data_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}_{tag.name}_positives.pkl")
# if cached_data_path.exists() and not overwrite:
#     print(f"Loaded data from: {cached_data_path}")
#     df = pd.read_pickle(cached_data_path)
    
# else:
#     all_df = load_data(datasets)
#     # positives_df = get_positives(all_df, tag)
#     df = get_positives(all_df, tag.name)
#     df = df[df["content_type"] == "text/html"]
#     print(f"Saved data at: {cached_data_path}")
#     df.to_pickle(cached_data_path)

# %%
# data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1765)_neg(4086)_intermediate.pkl"
# classified_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred_with_img.pkl"
predicted = False
if datasets[0] == 'train':
    predicted_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/train_df_pred_after_training(522031).pkl"
    not_predicted_nodes_data_path = "/data/GIT/node_classifier_with_imgs/train/processed_dedup.pkl"
    data_path = "/data/GIT/web-annotation-extractor/data/processed/train/dataset_pos(4319)_neg(13732)_intermediate.pkl"

if datasets[0] == 'develop':
    predicted_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred_after_training(178346).pkl"
    not_predicted_nodes_data_path = "/data/GIT/node_classifier_with_imgs/develop/processed_dedup.pkl"
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
rel_seg = RelSegmenter()

# %%
s = Segmenter()
# s.transform_regexes()

# %% [markdown]
# ## Train Model

# %%
# s.augment_company_names_with_training_data(df)
# s.transform_regexes()
# # saved_path = s.save_model()

# %%
# s.load_model(str(saved_path).split('/')[-1].split('.pkl')[0])
s.load_model("segmenter_trained-4971")

# %% [markdown]
# # Get the Positives only

# %%
# #? Get images that are positive
# img_indices = classified_df["node_text"][classified_df["node_text"].apply(lambda x: True if 'http' in x else False)].index
# classified_df = classified_df.loc[img_indices]
# classified_df = classified_df[classified_df["node_gt_tag"] != 'none']
# len(classified_df)

# %%
# Get positives
classified_df = classified_df[classified_df["node_gt_tag"] != 'none']
len(classified_df)

# %% [markdown]
# # Transform

# %%
# import multiprocessing as mp
# p = mp.Pool(mp.cpu_count())
# transformed_texts = []
# for transformed_text in p.imap(s.transform_texts, classified_df["node_text"], chunksize = 50):
#     transformed_texts.append(transformed_text)
# print(len(transformed_texts))
# classified_df["node_text_t"] = transformed_texts
# original_len = len(classified_df)
# # classified_df = classified_df.drop_duplicates("node_text_t")
# print(f"Removed duplicates: {original_len} -> {len(classified_df)} ({100*(len(classified_df)-original_len)/original_len:.1f} %)")

# %%
# # #? Transform and get only text from images
# import multiprocessing as mp
# p = mp.Pool(mp.cpu_count())
# transformed_texts = []
# img_indices = classified_df["node_text"][classified_df["node_text"].apply(lambda x: True if 'http' in x else False)].index
# for transformed_text in p.imap(s.transform_texts, classified_df["node_text"].loc[img_indices], chunksize = 50):
#     transformed_texts.append(transformed_text)
# print(len(transformed_texts))
# classified_df.loc[img_indices, "node_text_t_img"] = transformed_texts

# %%
# transformed_texts

# %% [markdown]
# # Match

# %% [markdown]
# ## REL

# %%
# classified_df = classified_df[classified_df['domain'].isin(['1820productions.com', 'olivehorse.com', 'walkerhamill.com'])]

mentions_dataset = {}
domains = set(classified_df['domain'])
for domain in tqdm(domains):
    mentions_dataset[domain] = rel_seg.get_mentions_dataset(classified_df[classified_df['domain'] == domain]['node_text'])
    classified_df.loc[classified_df['domain'] == domain, 'mentions_dataset'] = classified_df[classified_df['domain'] == domain]['node_text'].apply(lambda node_text: mentions_dataset[domain].get(node_text))
classified_df = classified_df.assign(mentions= rel_seg.get_only_org_mentions(classified_df["mentions_dataset"].values))

# %%
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 100, 'display.max_rows',200, 'display.min_rows',200)

# test_data[["url", "node_gt_text", "mentions", "node_text", "mentions_dataset"]]

# %%
classified_df["mentions_to_predict"] = classified_df["mentions"].apply(lambda x: [k for y in x for k in x][0] if x else '')

# %%
import multiprocessing as mp
p = mp.Pool(mp.cpu_count())
transformed_texts = []
for transformed_text in p.imap(s.transform_texts, classified_df["mentions_to_predict"], chunksize = 50):
    transformed_texts.append(transformed_text)
print(len(transformed_texts))
classified_df["node_text_t"] = transformed_texts

# %%
# pd.set_option('display.max_columns',20, 'display.max_colwidth', 200, 'display.max_rows',20, 'display.min_rows',20)

# classified_df

# %%
# #? Faster than one node at a time
r = {}
domains = set(classified_df['domain'])
for domain in tqdm(domains):
    r[domain] = rel_seg.predict_companies(classified_df[classified_df['domain'] == domain]['node_text']).dropna().values

# %%
classified_df['rel_predictions'] = classified_df['domain'].apply(lambda x: list(r.get(x)))


# %%
# #? In case the module doesn't produce matches
def convert_predicted_urls_to_matches(predictions):
    matches = []
    # matches.append({'company_id': predictions, 'matches':['']})
    for prediction in set(predictions):
        matches.append({'company_id': prediction, 'matches':['']})
    return matches
    
classified_df['rel_matches'] = classified_df['rel_predictions'].apply(convert_predicted_urls_to_matches)

# %%
classified_df["node_text_t"]

# %%
# #? It doesn't work because it removes the nodes that couldn't find information
# node_texts = classified_df['node_text_t']
# r = rel_seg.predict_companies(node_texts)
# print(len(node_texts))
# len(r.dropna())

# %%
# r = []
# node_texts = classified_df[classified_df['domain'] == 'hitt.com']['node_text_t']
# for node_text in node_texts:
#     r.append(rel_seg.predict_companies([node_text]))
# print(len(node_texts))
# len(pd.Series([x[0] for x in r]).dropna())

# %% [markdown]
# ## Gazetteer

# %%
# Original
p = mp.Pool(mp.cpu_count())
matches = []
for match in p.imap(s.find_companies, classified_df["node_text_t"], chunksize = 50):
    matches.append(match)
print(len(matches))
classified_df["gaz_matches"] = matches

# Match only on Images
# p = mp.Pool(mp.cpu_count())
# matches = []
# classified_df["node_text_t_img"] = classified_df["node_text_t_img"].fillna('')
# for match in p.imap(s.find_companies, classified_df["node_text_t_img"], chunksize = 50):
#     matches.append(match)
# print(len(matches))
# classified_df["gaz_matches"] = matches

# %%
classified_df["both_matches"] = classified_df.apply(lambda x: x["gaz_matches"] + x["rel_matches"], axis=1)

# %%
# classified_df["gaz_matches_len"] = classified_df["gaz_matches"].apply(len)
# classified_df.sort_values("gaz_matches_len")

# t = classified_df[["gaz_matches", "rel_matches"]]
# t['gaz_matches_len'] = t['gaz_matches'].apply(len)
# t['rel_matches_len'] = t['rel_matches'].apply(len)

# %% [markdown]
# ## Predictions

# %%
predited_df = classified_df.copy()

# predited_df["matches"] = predited_df["rel_matches"]
predited_df["matches"] = predited_df["gaz_matches"]
# predited_df["matches"] = predited_df["both_matches"]

predited_df = predited_df[["url", "matches"]]
predited_df = predited_df.groupby("url").agg(lambda x: x)
predited_df["matches"] = predited_df["matches"].apply(lambda row: [y for x in row for y in x if type(x) == list])

# %%
merge = df.set_index('url').join(predited_df).reset_index()
merge["matches"] = merge["matches"].fillna('').apply(list)

# %%
merge['predicted_tag'] = merge['matches']

# %%
# #? Get the gt companies
merge["gt_tag_with_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value')])
# merge["gt_tag_without_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# %% [markdown]
# # Evaluate

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %% [markdown]
# ## Gazetteer

# %% [markdown]
# ### Train

# %%
print(f"{dataset_name} - Gaz")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)


# %% [markdown]
# ### Train Gazetteer

# %%
if dataset_name =='train':
    s.remove_frequent_terms_with_training_metrics(domain_metrics, 0.3)
    saved_path = s.save_model()

# %% [markdown]
# ### Evaluate

# %% [markdown]
# ## REL

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %% [markdown]
# ## Both (Gazetteer on the images +REL on text) 

# %%
print(f"{dataset_name} - Both")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 2000, 'display.max_rows',200, 'display.min_rows',200)
domain_metrics

# %%
# companies_library_df = pd.DataFrame(s.companies_library)
# companies_library_df = companies_library_df.explode('regexes')
# companies_library_df
# print(f"BEFORE - Number of regexes in companies_library_df: {len(companies_library_df)}")
# regexes_to_remove = [' 123 money ', ' zvrs ']
# new_companies_library = companies_library_df[~companies_library_df['regexes'].isin(regexes_to_remove)]
# new_companies_library = new_companies_library.groupby(['company_id', 'company_name']).agg(lambda x: sorted(list(set(x))))
# new_companies_library.reset_index(inplace=True)
# new_companies_library.to_dict('records')
# # new_companies_library
# # companies_library_df["regexes"] = companies_library_df["regexes"].apply(lambda row: [x for x in row if x not in regexes_to_remove] )
# # companies_library_df = companies_library_df[companies_library_df["regexes"].apply(len) > 0 ]
# # print(f"AFTER - Number of regexes in companies_library_df: {len(companies_library_df)}")
# # self.companies_library = companies_library_df.to_dict('records')

# %%

# %%
# pd.DataFrame(s.companies_library)

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(merge[merge["PAST_CLIENT-gt_text_count"] > 0], gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 2000, 'display.max_rows',2, 'display.min_rows',2)
# domain_metrics[domain_metrics["FN"] > 1]

# %%
# positives_df[positives_df['domain'] == 'progress.com'].iloc[0]['html_t'].find('infor')
# positives_df[positives_df['domain'] == 'progress.com'].iloc[0]['url'][7038:]
positives_df[positives_df['domain'] == 'progress.com'].iloc[0]['url']

# %%
positives_df

# %%

positives_df[positives_df['domain'] == 'applause.com'].apply(lambda row: row if [x.get("text") for x in row["PAST_CLIENT"] if "Just Eat UK" == x] else None, axis=1).dropna()
