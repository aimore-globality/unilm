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

# %% [markdown]
# # Get the Positives only

# %%
# Get positives
classified_df = classified_df[classified_df["node_gt_tag"] != 'none']
len(classified_df)

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
# disambiguations

# %%
# # #? Get predictions for each disambiguation
classified_df["predictions"] = classified_df["disambiguations"].apply(lambda disambiguations: [item['prediction'] for item in disambiguations] )
# classified_df["predictions"] = classified_df["predictions"].apply(lambda disambiguations: [disambiguations[0]] if len(disambiguations) > 0 else disambiguations) #? By getting only the first item in the disambiguation there is a 3% abs variation 
classified_df["predictions"]

# %%
# # #? Convert predictions to company_id
classified_df['rel_predictions'] = classified_df['predictions'].apply(lambda row: [rel_seg.wiki_title_to_kc_mappings.get(x) for x in row])
# classified_df['rel_predictions']

# %%
# # #? Convert rel_predictions to matches
classified_df['rel_matches'] = classified_df.apply(lambda row: [{'company_id': row['rel_predictions'][x], 'matches':row["mentions"][x]['ngram']} for x in range(len(row['rel_predictions']))], axis=1)
# classified_df['rel_matches']

# %%
# # #? Remove empty company_id (companies that are not in our taxonomy) 
classified_df['rel_matches'] = classified_df['rel_matches'].apply(lambda row:[x for x in row if x['company_id']])
# classified_df['rel_matches']

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

# %% [markdown]
# # Evaluate

# %%
# df = pd.read_csv("/data/GIT/unilm/markuplm/notebooks/REL_NER/kc_wiki_csv_mapping.csv")
# mappings = df.dropna(subset=['wikipedia_url']).apply(lambda x: (x['wikipedia_url'].split('/')[-1], x['taxonomy_id']), axis=1)
# .set_index("title").to_dict()["taxonomy_id"]

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
domain_metrics_2 = get_reconciliations_metrics_for_all_domains(merge, gt_col="gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics_2)

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
# merge[["gt_tag_with_img", "predicted_tag", 'PAST_CLIENT']]

# %%
domain_metrics

# %%
domain_metrics.sort_values(['f1_adjusted', 'num_positives']).dropna(subset='f1')

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
