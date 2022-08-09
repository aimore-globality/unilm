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
import multiprocessing as mp
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
conn = rel_seg.mention_detector.wiki_db.db
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM wiki')
c.fetchall()

# %%
# c.execute("SELECT * FROM sqlite_master WHERE type= 'index';") 
# c.fetchall()

# %%
# s = Segmenter()
# # s.transform_regexes()

# %% [markdown]
# ## Train Model

# %%
# s.augment_company_names_with_training_data(df)
# # s.transform_regexes()
# # # saved_path = s.save_model()

# %%
# # s.load_model(str(saved_path).split('/')[-1].split('.pkl')[0])
# s.load_model("segmenter_trained-4971")

# %% [markdown]
# # Get the Positives only

# %%
# #? Get images that are positive
# img_indices = classified_df["node_text"][classified_df["node_text"].apply(lambda x: True if 'http' in x else False)].index
# classified_df = classified_df.loc[img_indices]
# classified_df = classified_df[classified_df["node_gt_tag"] != 'none']
# len(classified_df)

# %%
# # #? Get positives
classified_df = classified_df[classified_df["node_gt_tag"] != 'none']

# # #? Get positives with all images:
# positives_with_all_images_indices = list(classified_df["node_text"].apply(lambda x: x if 'http' in x else None).dropna().index | classified_df[classified_df["node_gt_tag"] != 'none'].index)
# classified_df = classified_df.loc[positives_with_all_images_indices]

len(classified_df)

# %% [markdown]
# # Transform

# %%
# # # #? Transform all nodes
# p = mp.Pool(mp.cpu_count())
# transformed_texts = []
# for transformed_text in p.imap(s.transform_texts, classified_df["node_text"], chunksize = 50):
#     transformed_texts.append(transformed_text)
# print(len(transformed_texts))
# classified_df["node_text_t"] = transformed_texts
# # # #? In case I want to remove the duplicates after the transformation
# # original_len = len(classified_df)
# # classified_df = classified_df.drop_duplicates("node_text_t")
# # print(f"Removed duplicates: {original_len} -> {len(classified_df)} ({100*(len(classified_df)-original_len)/original_len:.1f} %)")

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

# %% [markdown]
# # Match

# %% [markdown]
# ## REL

# %%
# # #? Find mentions in each node
mentions = rel_seg.get_mentions_dataset(classified_df['node_text'])
classified_df["mentions"] = pd.Series(mentions)
classified_df["mentions"] = classified_df["mentions"].fillna('').apply(list)

# %%
# # #? Disambiguate each mentiions
disambiguations = rel_seg.disambiguate(classified_df["mentions"])
classified_df["disambiguations"] = pd.Series(disambiguations)
classified_df["disambiguations"] = classified_df["disambiguations"].fillna('').apply(list)

# %%
# # #? Get predictions for each disambiguation
classified_df["predictions"] = classified_df["disambiguations"].apply(lambda disambiguations: [item['prediction'] for item in disambiguations] )
# classified_df["predictions"] = classified_df["predictions"].apply(lambda disambiguations: [disambiguations[0]] if len(disambiguations) > 0 else disambiguations) #? By getting only the first item in the disambiguation there is a 3% abs variation 

# # #? Convert predictions to company_id
classified_df['rel_predictions'] = classified_df['predictions'].apply(lambda row: [rel_seg.wiki_title_to_kc_mappings.get(x) for x in row])

# # #? Convert rel_predictions to matches
classified_df['rel_matches'] = classified_df.apply(lambda row: [{'company_id': row['rel_predictions'][x], 'matches':row["mentions"][x]['ngram']} for x in range(len(row['rel_predictions']))], axis=1)

# # #? Remove empty company_id (companies that are not in our taxonomy) 
classified_df['rel_matches'] = classified_df['rel_matches'].apply(lambda row:[x for x in row if x['company_id']])

# %% [markdown]
# ## Intermediate Experiment 
# Check for how many gt_text the mention detection can detect?   

# %%
# def get_the_matches():
#     return classified_df['rel_matches'][classified_df['rel_matches'].apply(len) > 0].apply(lambda x: [y['matches'] for y in x])

# classified_df['matches_found_by_rel'] = get_the_matches()
# matches_found_by_rel = pd.Series(classified_df.explode('matches_found_by_rel')['matches_found_by_rel'].dropna().values).apply(lambda x: x)


# print(f"Total number of companies in node_gt_text: {len(set(node_gt_text))}")
# print(f"Total number of companies in matches_found_by_rel: {len(set(matches_found_by_rel))}")
# print(f"Difference of node_gt_text and matches_found_by_rel: {len(set(matches_found_by_rel) - set(node_gt_text))}")
# print(f"Difference of matches_found_by_rel and node_gt_text: {len(set(node_gt_text) - set(matches_found_by_rel))}" )
# print(f"Intersection of node_gt_text and matches_found_by_rel: {100*len(set(node_gt_text) & set(matches_found_by_rel)) / len(set(node_gt_text)):.1f} %")
# diff = set(node_gt_text) - set(matches_found_by_rel)
# print(f"difference: {diff}")

# %%
# mentions_found_by_rel = set(classified_df["mentions"].apply(lambda row: [x["ngram"] for x in row]).explode().dropna().values)
# node_gt_text = set(pd.Series(classified_df.explode('node_gt_text')['node_gt_text'].dropna().values).apply(lambda x: x if 'http' not in x else None).dropna())

# print(f"Unique number of companies in node_gt_text: {len(node_gt_text)}")
# print(f"Unique number of mentions by rel: {len(mentions_found_by_rel)}")

# print(f"Intersection between node_gt_text and mentions_found_by_rel: {len(mentions_found_by_rel & node_gt_text)}")
# diff = node_gt_text - mentions_found_by_rel
# print(f"Difference of node_gt_text and mentions_found_by_rel: {len(diff)}" )
# print(f"Difference of mentions_found_by_rel and node_gt_text: {len(mentions_found_by_rel - node_gt_text)}")
# # print(f"Intersection of node_gt_text and matches_found_by_rel: {100*len(set(node_gt_text) & set(matches_found_by_rel)) / len(set(node_gt_text)):.1f} %")

# %%
# diff_df = pd.DataFrame(list(diff)).dropna()
# diff_df

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

# %%
from new_segmenter import Transform
import numpy as np

t = Transform(transformations=["decode", "lower", "replace_ampersand", "replace_symbols", "remove_symbols", "normalize_any_space"])
def clean(string):
    string = t.transform(string)
    return string

classified_df['mapping'] = classified_df.apply(lambda row: {x['text']:x['value'] for x in row["PAST_CLIENT-annotations"]}, axis=1) #? Create a mapping from gt_text to gt_value to be used on the nodes
classified_df['node_gt_value'] = classified_df.apply(lambda row: [row['mapping'].get(x) for x in row["node_gt_text"] if row['mapping'].get(x) and 'http' not in x], axis=1) #? Apply this mapping and get gt_value per node

sub_classified_df = classified_df[classified_df['node_gt_value'].apply(len) > 0] #? Select nodes that only have gt_value 

gt_value_correct_indices = sub_classified_df[sub_classified_df.apply(lambda row: np.any([True if x in row["PAST_CLIENT-gt_value"] else False for x in row["rel_predictions"]]), axis=1)].index #? Get indices where gt_value is correct 
gt_text_correct_indices = sub_classified_df[sub_classified_df.apply(lambda row: clean(row['node_gt_text'][0]) in [clean(x['ngram']) for x in row['mentions']], axis=1)].index #? Get indices where cleaned gt_text is in cleaned mentions  

correct_indices = list(set(gt_text_correct_indices) | set(gt_value_correct_indices))

gt_text_in_rel_mentions = sub_classified_df[sub_classified_df.index.isin(correct_indices)].apply(lambda row: (row['domain'], row['url'], row['node_text'], row['node_gt_text'][0], [x['ngram'] for x in row['mentions']], [x['candidates'] for x in row['mentions']]), axis=1) #? Get gt_text_in_rel_mentions
gt_text_in_rel_mentions_df = pd.DataFrame(list(gt_text_in_rel_mentions.values), columns = ['domain', 'url', 'node_text', 'node_gt_text', 'ngram', 'candidates'])
gt_text_in_rel_mentions_df["ngram_len"] = gt_text_in_rel_mentions_df["ngram"].apply(len)

gt_text_not_in_rel_mentions = sub_classified_df[~sub_classified_df.index.isin(correct_indices)].apply(lambda row: (row['domain'], row['url'], row['node_text'], row['node_gt_text'][0], [x['ngram'] for x in row['mentions']], [x['candidates'] for x in row['mentions']]), axis=1) #? Get gt_text_not_in_rel_mentions
gt_text_not_in_rel_mentions_df = pd.DataFrame(list(gt_text_not_in_rel_mentions.values), columns = ['domain', 'url', 'node_text', 'node_gt_text', 'ngram', 'candidates'])
gt_text_not_in_rel_mentions_df["ngram_len"] = gt_text_not_in_rel_mentions_df["ngram"].apply(len)

# %%
len(set(sub_classified_df["node_gt_text"].explode()))

# %%
gt_text_not_in_rel_mentions_df

# %%
gt_text_in_rel_mentions_df

# %%
print("LARGE NER with company filtered - drop duplicates within a domain")
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))
not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))

gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({in_size})_covered_NER_LARGE_with_cand_filtered.html")
gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({not_in_size})_NOT_covered_NER_LARGE_with_cand_filtered.html")

total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
print("Fast NER with company filtered - drop duplicates within a domain")
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))
not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))

# gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered_NER_LARGE_with_cand_filtered.html")
# gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered_NER_LARGE_with_cand_filtered.html")

total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
print("Fast NER WITHOUT company filtered - drop duplicates within a domain")
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))
not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))

# gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered_NER_LARGE_with_cand_filtered.html")
# gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered_NER_LARGE_with_cand_filtered.html")

total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered_NER_LARGE_with_cand_filtered.html")

not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered_NER_LARGE_with_cand_filtered.html")
total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered_Ontonotes_with_cand_filtered.html")

not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered_Ontonotes_with_cand_filtered.html")
total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered.html")

not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered.html")
total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({in_size})_covered.html")

not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text'))
gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len').to_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered.html")
total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
gt_text_not_in_rel_mentions_df['ngram_len'] = gt_text_not_in_rel_mentions_df['ngram'].apply(len)
gt_text_not_in_rel_mentions_df.drop_duplicates('node_gt_text').sort_values('ngram_len')

# %%
gt_text_not_in_rel_mentions_df

# %%
df = pd.read_html(f"Analyse_REL_MD_recall_dedup({not_in_size})_NOT_covered.html")
df[0]

# %%
# sent = "AMAZON"
# sent = "Skanska, Costain and STRABAG Joint Venture (SCS JV)"
sent = "BT"
t = classified_df[classified_df["node_text"].apply(lambda x: sent in x) ]["node_text"]
t_df = pd.DataFrame(t)
t_df['mentions'] = pd.Series(rel_seg.get_mentions_dataset(t))
t_df

# %%
t
rel_seg.get_mentions_dataset(pd.Series('accenture'))

# %% [markdown]
# ## Gazetteer

# %%
# # # #? Match all transformed nodes
# p = mp.Pool(mp.cpu_count())
# matches = []
# for match in p.imap(s.find_companies, classified_df["node_text_t"], chunksize = 50):
#     matches.append(match)
# print(len(matches))
# classified_df["gaz_matches"] = matches

# # # #? Match only on Images
# # p = mp.Pool(mp.cpu_count())
# # matches = []
# # classified_df["node_text_t_img"] = classified_df["node_text_t_img"].fillna('')
# # for match in p.imap(s.find_companies, classified_df["node_text_t_img"], chunksize = 50):
# #     matches.append(match)
# # print(len(matches))
# # classified_df["gaz_matches"] = matches

# %% [markdown]
# # Merge Both

# %%
# classified_df["both_matches"] = classified_df.apply(lambda x: x["gaz_matches"] + x["rel_matches"], axis=1)

# %%
# classified_df["gaz_matches_len"] = classified_df["gaz_matches"].apply(len)
# classified_df.sort_values("gaz_matches_len")

# t = classified_df[["gaz_matches", "rel_matches"]]
# t['gaz_matches_len'] = t['gaz_matches'].apply(len)
# t['rel_matches_len'] = t['rel_matches'].apply(len)

# %%
# classified_df.loc[224856][["disambiguations", "mentions"]]
# classified_df

# %% [markdown]
# ## Predictions

# %%
predited_df = classified_df.copy()

predited_df["matches"] = predited_df["rel_matches"]
# predited_df["matches"] = predited_df["gaz_matches"]
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
# # #? Get the gt companies
merge["gt_tag_with_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value')])
# merge["gt_tag_without_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# %%
# # pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',200, 'display.min_rows',200)
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 100, 'display.max_rows',4, 'display.min_rows',4)

# merge

# %% [markdown]
# # Evaluate

# %%
# merge[merge['domain'] == 'palisade.com']['matches'].apply(lambda row: [x for x in row if 'Brasil' in x['matches']])
# merge.loc[1648:1648]

# for y in predited_df.loc[merge.loc[1648:1648].url.values[0]]:
#     print([(x['company_id'], x['company_id']) for x in y if x['matches'] == 'Brasil'])

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
domain_metrics.iloc[-1:][["TP_pred","FP_pred","FP_seg"]]

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL") # Tried again with Fast NER and candidates filtered
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL") # Tried again
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

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

# %%
print(f"{dataset_name} - Gaz")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

domain_metrics.to_csv(f"/data/GIT/unilm/markuplm/notebooks/results_reconciliations/{dataset_name}-Gaz.csv")

# %% [markdown]
# ## REL

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

domain_metrics.to_csv(f"/data/GIT/unilm/markuplm/notebooks/results_reconciliations/{dataset_name}-REL.csv")

# %%
print(f"{dataset_name} - REL")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

domain_metrics.to_csv(f"/data/GIT/unilm/markuplm/notebooks/results_reconciliations/{dataset_name}-REL.csv")

# %% [markdown]
# ## Both (Gazetteer on the images +REL on text) 

# %%
print(f"{dataset_name} - Both")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

domain_metrics.to_csv(f"/data/GIT/unilm/markuplm/notebooks/results_reconciliations/{dataset_name}-REL_and_Gaz.csv")

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
