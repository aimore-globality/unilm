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
#     display_name: Python 3.9.12 ('wae39')
#     language: python
#     name: python3
# ---

# %%
from new_segmenter import Segmenter
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.evaluations.metric_functions import get_reconciliations_metrics_for_all_domains, calculate_metrics_for_dataset

# from web_annotation_extractor.common.utils.general_utils import deserialize_annotations
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
#     positives_df = pd.read_pickle(cached_data_path)
    
# else:
#     all_df = load_data(datasets)
#     # positives_df = get_positives(all_df, tag)
#     positives_df = get_positives(all_df, tag.name)
#     positives_df = positives_df[positives_df["content_type"] == "text/html"]
#     print(f"Saved data at: {cached_data_path}")
#     positives_df.to_pickle(cached_data_path)

# %%
data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1830)_neg(4587)_intermediate.pkl"
# data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1765)_neg(4086)_intermediate.pkl"
df = pd.read_pickle(data_path)
df = df.rename(columns={"PAST_CLIENT-annotations": "PAST_CLIENT"})
print(len(df))
classified_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred_with_img.pkl"
classified_df = pd.read_pickle(classified_nodes_data_path)
print(len(classified_df))

# %%
# set(df['domain']) & set(classified_df['domain']) 
classified_df

# %% [markdown]
# # Load Model

# %%
s = Segmenter()
s.transform_regexes()

# %% [markdown]
# ## Train Model

# %%
company_id_company_name_and_regexes = s.get_company_id_and_regexes_from_annotations(df)
s.augment_library_with_training_data(company_id_company_name_and_regexes)
s.transform_regexes()
saved_path = s.save_model()

# %%
s.load_model(str(saved_path).split('/')[-1].split('.pkl')[0])
# s.load_model("segmenter_trained-6316")

# %% [markdown]
# # Transform

# %%
import multiprocessing as mp
p = mp.Pool(mp.cpu_count())
transformed_texts = []
for transformed_text in p.imap(s.transform_texts, classified_df["node_text"]):
    transformed_texts.append(transformed_text)
print(len(transformed_texts))
classified_df["node_text_t"] = transformed_texts

# %% [markdown]
# # Match

# %%
p = mp.Pool(mp.cpu_count())
matches = []
for match in p.imap(s.find_companies, classified_df["node_text_t"]):
    matches.append(match)
print(len(matches))
classified_df["matches"] = matches

# %%
classified_df = classified_df[["url", "matches"]]
classified_df = classified_df.groupby("url").agg(lambda x: x)
classified_df["matches"] = classified_df["matches"].apply(lambda row: [y for x in row for y in x])

# %%
merge = df.set_index('url').join(classified_df).reset_index()
merge["matches"] = merge["matches"].fillna('').apply(list)

# %%
len(merge)

# %%
# #? Extract just the company_id from matches
merge['predicted_tag'] = merge['matches']

# %%
# #? Get the gt companies
merge["gt_tag_with_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value')])
merge["gt_tag_without_img"] = merge['PAST_CLIENT'].apply(lambda row: [str(x.get('value')) for x in row if x.get('value') and 'http' not in x.get('text')])

# %% [markdown]
# # Evaluate

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col = "gt_tag_with_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(merge, gt_col = "gt_tag_without_img", predicted_col="predicted_tag", annotations_col='PAST_CLIENT', negative_percentage=0.1)
calculate_metrics_for_dataset(domain_metrics)

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 2000, 'display.max_rows',2, 'display.min_rows',2)
domain_metrics[domain_metrics["TP"] > 1]["predicted_tag"].index

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
