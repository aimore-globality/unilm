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
datasets = ['train']
# datasets = ['develop']

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

# %% [markdown]
# ### MyNER

# %%
from REL_NER.my_ner import MyNER
ner = MyNER()

# %% [markdown]
# ### MyREL

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

# %% [markdown]
# ### NewGazetteer

# %%
s = Segmenter()
print(s.number_of_companies(), s.number_of_regexes())
# s.number_of_companies()
# s.number_of_regexes()

# %%
pd.DataFrame(s.companies_library)

# %%
s.augment_company_names_with_training_data(df)
print(s.number_of_companies(), s.number_of_regexes())

s.transform_regexes() 
print(s.number_of_companies(), s.number_of_regexes())

# s.remove_duplicated_regexes_and_sort()
# print(s.number_of_companies(), s.number_of_regexes())

# %%
s.augment_company_names_with_prior_wiki_db()
print(s.number_of_companies(), s.number_of_regexes())

s.transform_regexes()
print(s.number_of_companies(), s.number_of_regexes())

s.remove_duplicated_regexes_and_sort()
print(s.number_of_companies(), s.number_of_regexes())

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)
pd.DataFrame(s.companies_library).explode('regexes')

# %%
# saved_path = s.save_model()

# %%
# # s.load_model(str(saved_path).split('/')[-1].split('.pkl')[0])
# # s.load_model("segmenter_trained-5242")
# s.load_model("segmenter_trained-63600")

# print(s.number_of_companies(), s.number_of_regexes())

# %% [markdown]
# # Get the Positives only

# %%
# #? Get images that are positive
# img_indices = classified_df["node_text"][classified_df["node_text"].apply(lambda x: True if 'http' in x else False)].index
# classified_df = classified_df.loc[img_indices]
# classified_df = classified_df[classified_df["node_gt_tag"] != 'none']
# len(classified_df)

# %%
# # #? Get positives that don't contain images:
# classified_df = classified_df[classified_df["node_gt_tag"] != 'none'].apply(lambda x: x if 'http' not in x else None).dropna().index & classified_df[classified_df["node_gt_tag"] != 'none'].index)
positives_with_no_images_indices = list(classified_df["node_text"].apply(lambda x: x if 'http' not in x else None).dropna().index & classified_df[classified_df["node_gt_tag"] != 'none'].index)
classified_df = classified_df.loc[positives_with_no_images_indices]

# # #? Get positives and all nodes that are images:
# positives_with_all_images_indices = list(classified_df["node_text"].apply(lambda x: x if 'http' in x else None).dropna().index | classified_df[classified_df["node_gt_tag"] != 'none'].index)
# classified_df = classified_df.loc[positives_with_all_images_indices]

len(classified_df)

# %%
classified_df

# %% [markdown]
# # Find Mentions

# %% [markdown]
# ## Flair NER

# %%
flair_ner_sentence = classified_df['node_text'].apply(lambda row: ner.format_sentences(ner.predict([row])))
classified_df['flair_ner_sentence'] = flair_ner_sentence

# %%
classified_df.reset_index(inplace=True)

# %%
sentences_p_exploded = classified_df.explode('flair_ner_sentence')

# %%
sentences_p_exploded['flair_ner_sentence'] = sentences_p_exploded['flair_ner_sentence'].fillna('')
sentences_p_exploded['node_text_flair_ner'] = sentences_p_exploded['flair_ner_sentence'].apply(lambda x: x[0] if x else '').values

# %% [markdown]
# ## Spacy NER

# %%
# import spacy

# spacy.prefer_gpu()
# spacy_ner = spacy.load("en_core_web_trf")

# %%
# spacy_ner_sentence = classified_df['node_text'].apply(lambda row: spacy_ner(row))
# classified_df['spacy_ner_sentence'] = spacy_ner_sentence

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
# # #? In case I want to remove the duplicates after the transformation
# original_len = len(classified_df)
# classified_df = classified_df.drop_duplicates("node_text_t")
# print(f"Removed duplicates: {original_len} -> {len(classified_df)} ({100*(len(classified_df)-original_len)/original_len:.1f} %)")

# %%
# # #? Transform all mentions from NER
p = mp.Pool(mp.cpu_count())
transformed_texts = []
for transformed_text in p.imap(s.transform_texts, sentences_p_exploded["node_text_flair_ner"], chunksize = 50):
    transformed_texts.append(transformed_text)
print(len(transformed_texts))
sentences_p_exploded["node_text_flair_ner_t"] = transformed_texts
sentences_p_exploded["node_text_flair_ner_t"] = sentences_p_exploded["node_text_flair_ner_t"].fillna('')

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
import numpy as np
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',50, 'display.min_rows',50)
classified_df[classified_df.apply(lambda row: np.any([True for x in row["node_gt_text"] if 'Vodafone' in x]), axis=1)]

# %%
len(set(sub_classified_df["node_gt_text"].explode()))

# %%
gt_text_not_in_rel_mentions_df

# %%
gt_text_in_rel_mentions_df

# %%
print("FAST LOWER NER without company filtered - drop duplicates within a domain")
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))
not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))

gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({in_size})_covered_NER_FAST_LOWER_without_cand_filtered.html")
gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({not_in_size})_NOT_covered_NER_FAST_LOWER_without_cand_filtered.html")
``
total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

# %%
print("LARGE NER without company filtered - drop duplicates within a domain")
in_size = len(gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))
not_in_size = len(gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']))

gt_text_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({in_size})_covered_NER_LARGE_without_cand_filtered.html")
gt_text_not_in_rel_mentions_df.drop_duplicates(['node_gt_text', 'domain']).sort_values('ngram_len').to_html(f"Analyse_REL_MD_coverage_domain_dedup({not_in_size})_NOT_covered_NER_LARGE_without_cand_filtered.html")

total = in_size + not_in_size
print(f"Unique gt_text: Total: {total} \n REL matched: {in_size} ({100*in_size/total:.2f}%) |  REL couldn't match: {not_in_size} ({100*not_in_size/total:.2f}%)")

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
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
# pd.DataFrame(s.companies_library)

# %%
# # #? Match all transformed nodes
p = mp.Pool(mp.cpu_count())
matches = []
for match in p.imap(s.find_companies, classified_df["node_text_t"], chunksize = 50):
    matches.append(match)
print(len(matches))
classified_df["gaz_matches"] = matches

# # # #? Match only on Images
# # p = mp.Pool(mp.cpu_count())
# # matches = []
# # classified_df["node_text_t_img"] = classified_df["node_text_t_img"].fillna('')
# # for match in p.imap(s.find_companies, classified_df["node_text_t_img"], chunksize = 50):
# #     matches.append(match)
# # print(len(matches))
# # classified_df["gaz_matches"] = matches

# %%
# # #? Match all transformed node_text_ner_t
p = mp.Pool(mp.cpu_count())
matches = []
for match in p.imap(s.find_companies, sentences_p_exploded["node_text_flair_ner_t"], chunksize = 50):
    matches.append(match)
print(len(matches))
sentences_p_exploded['gaz_matches'] = matches

classified_df["flair_ner_t_gaz_matches"] = sentences_p_exploded.groupby("level_0")["gaz_matches"].agg(list).apply(lambda row: [x for x in row if len(x) > 0]).reset_index()['gaz_matches'].apply(lambda x: x[0] if len(x) > 0 else x)

# %%
# [x for x in s.companies_library if x['company_id'] in ["http://graph.globality.io/platform/KnownCompany#enterprise_holdings"]]
# [x for x in s.companies_library if x['company_id'] in ["http://graph.globality.io/platform/KnownCompany#mccormick_company", "http://graph.globality.io/platform/KnownCompany#mccormick"]]

# %%
# sentences_p_exploded["gaz_matches"][sentences_p_exploded["gaz_matches"].apply(len) > 0]

# %%
# sentences_p_exploded.groupby("level_0")["gaz_matches"].agg(list).apply(lambda row: [x for x in row if len(x) > 0]).reset_index()['gaz_matches'].apply(lambda x: x[0] if len(x) > 0 else x)

# %%
# classified_df["flair_ner_t_gaz_matches"] = sentences_p_exploded.groupby("level_0")["gaz_matches"].agg(list).apply(lambda row: [x for x in row if len(x) > 0]).reset_index()['gaz_matches'].apply(lambda x: x[0] if len(x) > 0 else x)
# classified_df["ner_gaz_matches"] = sentences_p_exploded.groupby("level_0")["gaz_matches"].agg(list).apply(lambda row: row[0] if len(row) > 0 else []).reset_index()['gaz_matches']

# %%
# # pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
# classified_df[["gaz_matches", "ner_gaz_matches"]]

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

# %%
# classified_df["mentions_len"] = classified_df["mentions"].apply(len)
# classified_df["disambiguations_len"] = classified_df["disambiguations"].apply(len)
# classified_df.sort_values("mentions_len")
# classified_df[classified_df["disambiguations_len"] > 0 ]

# %%
# classified_df[classified_df["rel_predictions"].apply(len) > 0 ]['disambiguations'].values

# %% [markdown]
# # Predictions

# %%
predited_df = classified_df.copy()

# predited_df["matches"] = predited_df["rel_matches"]
predited_df["matches"] = predited_df["gaz_matches"]
# predited_df["matches"] = predited_df["flair_ner_t_gaz_matches"]
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
merge["gt_tag_without_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# %% [markdown]
# # Evaluate

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
GAZ, with training data augmentation

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - NER")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - NER")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - NER + GAZ")
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
print(f"{dataset_name} - GAZ")
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
s.number_of_companies(), s.number_of_regexes()

# %%
if dataset_name =='train':
    # s.remove_frequent_terms_with_training_metrics(domain_metrics, 0.3)
    s.remove_frequent_terms_with_training_metrics(domain_metrics, 0.99, save_df=True)
    
    # saved_path = s.save_model()

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

# %% [markdown]
# # Error Analysis

# %% [markdown]
# ## FN Error Analysis

# %% [markdown]
# ## FP Error Analysis

# %%
df["PAST_CLIENT-gt_text"]

# %%
# classified_df[classified_df['node_gt_text'].apply(lambda row: row == ['3M'])]
classified_df[classified_df['domain'] == 'hawthorneadvertising.com']["url"]
# ["node_text"]

# %%
# classified_df[classified_df['node_gt_text'].apply(lambda row: row == ['3M'])]
classified_df[classified_df['domain'] == 'sailpoint.com']["url"]
# ["node_text"]

# %%
classified_df[classified_df['domain'] == domain]["PAST_CLIENT-gt_value"]

# %%
domain_FN_company_ids

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
domain_nodes[{"node_gt_value", "PAST_CLIENT-gt_value"}]

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

sorted(classified_df[classified_df['domain'] == domain]["node_gt_value"].explode().dropna().values)

# %%
sorted(fn_error[fn_error['domain'] == domain]["gt_tag_without_img"].iloc[0])
# domain

# %%
# classified_df[classified_df['domain'] == domain]
classified_df['mapping'] = classified_df.apply(lambda row: {x['text']:x['value'] for x in row["PAST_CLIENT-annotations"]}, axis=1) #? Create a mapping from gt_text to gt_value to be used on the nodes
classified_df['node_gt_value'] = classified_df.apply(lambda row: [row['mapping'].get(x) for x in row["node_gt_text"] if row['mapping'].get(x) and 'http' not in x], axis=1) #? Apply this mapping and get gt_value per node

# %%
# domain_FN_company_ids
classified_df[classified_df["node_text_t"].apply(lambda x: 'http' in x)]["node_text_t"]

# %%

# #! I want to see the FN segmentations that the model missed, and what are the regexes of these companies that the model couldn't find
domain_metrics = domain_metrics.reset_index()
domain_metrics = domain_metrics.rename({'index':"domain"},axis=1)
fn_error = domain_metrics[domain_metrics["FN"] > 0].sort_values("FN",ascending=False).explode("FN_seg").sort_values("FN_seg")
mapping_compId_regexes = pd.DataFrame(s.companies_library).set_index("company_id")["regexes"].to_dict()
fn_error["regexes"] = fn_error["FN_seg"].apply(lambda row: mapping_compId_regexes.get(row[0]))
fn_error

# %%

# %%

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

# %%
df

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 100, 'display.max_rows',10, 'display.min_rows',10)
# pd.Series(classified_df[classified_df['url'].isin(pages_that_contain_gt_value_in_page_but_not_in_node_dataset)]['PAST_CLIENT-gt_value_untax'].explode().dropna()).value_counts()

# classified_df['mapping'] = classified_df.apply(lambda row: {x['text']:x['value'] for x in row["PAST_CLIENT-annotations"]}, axis=1) #? Create a mapping from gt_text to gt_value to be used on the nodes
# classified_df['node_gt_value'] = classified_df.apply(lambda row: [row['mapping'].get(x) for x in row["node_gt_text"] if row['mapping'].get(x) and 'http' not in x], axis=1) #? Apply this mapping and get gt_value per node

# classified_df[classified_df['url'].isin(pages_that_contain_gt_value_in_page_but_not_in_node_dataset)].explode('node_gt_text').dropna()[["node_text"]]


# %%
# #! Try to convert these pages into nodes
from lxml import etree
import lxml
htmls = df[df['url'].isin(pages_that_contain_gt_value_in_page_but_not_in_node_dataset)]["html"]

# %%
dom_tree = etree.ElementTree(lxml.html.fromstring(htmls.iloc[0]))

# %%
import collections
matched_xpaths = []  # The resulting list of xpaths to be returned.
current_xpath_data = dict()  # The resulting dictionary to save all page data.

gt_text_in_nodes = dict()  # A list of the gt_text in each xpath node

overall_xpath_dict = collections.defaultdict(set)

current_page_nodes_in_order = []
is_truth_value_list = []
min_node_text_size = 0
max_node_text_size = 100_000_000
for node in dom_tree.iter():
    # The value can only be matched in the text of the node or the tail.
    node_text_dict = {
        "node_text": node.text,
        "node_tail_text": node.tail,
    }  # ?The only nodes that are considered here are the node.text and node.tail

    for text_part_flag, node_text in node_text_dict.items():
        if node_text:
            if (
                node.tag != "script"
                and "javascript" not in node.attrib.get("type", "")
                and min_node_text_size <= len(node_text.strip()) < max_node_text_size
            ):  #! Remove java/script and min_node_text # TODO (Aimore): Make this comparisons more explicity and descriptive
                # """Matches the ground truth value with a specific node in the domtree.

                node_attribute = node.attrib.get("type", "")
                node_tag = node.tag
                node_text_split = node_text.split("--BRRB--")
                len_brs = len(node_text_split)  # The number of the <br>s.
                for index, etext in enumerate(node_text_split):

                    if text_part_flag == "node_text":
                        xpath = dom_tree.getpath(node)

                    elif text_part_flag == "node_tail_text":
                        xpath = dom_tree.getpath(node) + "/tail"

                    if len_brs >= 2:
                        xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]

                    # clean_etext = clean_spaces(etext)
                    clean_etext = etext

                    # ? Update the dictionary.
                    current_xpath_data[xpath] = clean_etext
                    overall_xpath_dict[xpath].add(clean_etext)
                    current_page_nodes_in_order.append(
                        (clean_etext, xpath, node_attribute, node_tag)
                    )

                    # ? Clean the groundtruth and the node text. Check if the groundtruth is in the node text.
                    # clean_etext = clean_format_str(clean_etext)

                    # ? Create node ground truth by checking if the the gt_text is in the clean node_text
                    # gt_text_in_node = []
                    # for gt_value in clean_gt_values:
                    #     if f" {gt_value.strip()} ".lower() in f" {clean_etext.strip()} ".lower():
                    #         gt_text_in_node.append(gt_value)
                    #         matched_xpaths.append(xpath)
                    #         is_truth_value_list.append(
                    #             len(current_page_nodes_in_order) - 1
                    #         )
                    #         # break #! I am not sure why Iadded this break, I'm commenting it because I think all gt_values should be added in a node

                    # if len(matched_xpaths) == 0:
                    #     gt_text_in_nodes[xpath] = []
                    # else:
                    #     gt_text_in_nodes[xpath] = gt_text_in_node

# %%
# pd.DataFrame(overall_xpath_dict.items())[1][pd.DataFrame(overall_xpath_dict.items())[1].apply(lambda row: any(['3m' in x.lower() for x  in list(row)]))]
# #? Some of the reasons that these urls don't appear in the node_dataset:
#? 1. Annotation from the page cannot be found in the nodes
#? 2. Nodes that are very small (2 char) and very long (10_000 char) are droppped
#? 3. Pages that are very long are droppped
#? 4. The version of the page doesn't exists in the node_dataset because  it was dropppend 

# %%
domain_pages[domain_pages["PAST_CLIENT-gt_value"].apply(lambda row: 'http://graph.globality.io/platform/KnownCompany#standard_life' in row)]

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

for enum, domain in enumerate(sorted(list((set(fn_error['domain']))))):
    domain_FN_company_ids = fn_error[fn_error['domain'] == domain]['FN_pred'].iloc[0]
    domain_nodes = classified_df[classified_df['domain'] == domain]
    fn_nodes = domain_nodes[domain_nodes["node_gt_value"].apply(lambda row: any([True if x in domain_FN_company_ids else False for x in row]))]
    print(f'Domain: {domain}, Nodes: {len(domain_nodes)}. FN companies in domain:\n {domain_FN_company_ids}')
    display(fn_nodes[["node_gt_value", "node_gt_text", "url", "domain"]])
    if enum==3:
        break

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

for enum, domain in enumerate(sorted(list((set(fn_error['domain']))))):
    domain_FN_company_ids = fn_error[fn_error['domain'] == domain]['FN_pred'].iloc[0]
    domain_pages = df[df['domain'] == domain]
    fn_pages = domain_pages[domain_pages["PAST_CLIENT-gt_value"].apply(lambda row: any([True if x in domain_FN_company_ids else False for x in row]))]
    print(f'Domain: {domain}, Pages: {len(domain_pages)}. FN pages in domain:\n {domain_FN_company_ids}')
    display(fn_pages[["url","domain","html","",""]])
    if enum==3:
        break

# %%

# #! I want to see all the regexes and companies that contributed to the FP for a specific domain
pd.DataFrame(s.companies_library).explode('regexes')[pd.DataFrame(s.companies_library).explode('regexes')['regexes'].isin(domain_metrics.loc["gorkana.com"]['FP_seg'])]

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
