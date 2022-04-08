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
#     display_name: Python 3.8.12 ('wae_test')
#     language: python
#     name: python3
# ---

# %%
from microcosm_sagemaker.bundle import BundleInputArtifact
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.common.utils.parallel_tools import OptimalParallel
from web_annotation_extractor.bundles.past_client.bundle import PastClientBundle
from web_annotation_extractor.bundles.past_client.segmentation import *
from web_annotation_extractor.evaluations.metric_functions import calculate_metrics_for_dataset, get_reconciliations_metrics_for_all_domains, average_domains_metrics
from web_annotation_extractor.evaluations.visualization_functions import plot_performance
from web_annotation_extractor.evaluations.metric_functions import combine_and_get_sorted_list
from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter

graph = create_object_graph('test')
pd.set_option('max_columns',60, 'max_colwidth',80, 'max_rows',5)

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

known_company_names = pd.DataFrame([company.name for company in known_company_taxonomy])
known_company_names_taxonomy = pd.DataFrame([(company, company.name) for company in known_company_taxonomy], columns=["taxonomy_id", "company_name"])

# %%
known_company_names.value_counts()

# %%
known_company_names_taxonomy.to_html("known_company_names_taxonomy.html")

# %%
company.

# %% [markdown]
# # Load Data

# %%
dataset = 'develop'
print(dataset)
if dataset == 'develop':
    data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1735)_neg(4032)_intermediate.pkl"
df = pd.read_pickle(data_path)

print(len(df))
df = df.set_index("url")
df.head(2)

# %%
gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
gazetteer.config["stop_words"] = True
gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
gazetteer.prepare_to_segment()

# %%
print(len(gazetteer.segmenter))

# %% [markdown]
# ## Similar Company Names

# %%
gazetteer_df = pd.DataFrame(gazetteer.segmenter, columns=["company_name", "company_regex"])

above_one = gazetteer_df[~gazetteer_df["company_regex"].isin([' bank of ', ' university of '])]["company_regex"].value_counts()[gazetteer_df[~gazetteer_df["company_regex"].isin([' bank of ', ' university of '])]["company_regex"].value_counts() > 1]

similar_company_names = gazetteer_df[(gazetteer_df["company_regex"].isin(above_one.index)) & ~(gazetteer_df["company_regex"].isin([" banca "])) ].sort_values("company_regex")
similar_company_names


# %%
similar_company_names.to_html("similar_company_names.html")

# %%
df.head(3)

# %%
for d_name, d in df.groupby("domain").aggregate("PAST_CLIENT-annotations"):
    d

# %%
from web_annotation_extractor.evaluations.metric_functions import *

gt_text_value = pd.DataFrame([y for x in d for y in x if y.get('value') is not None ])
gt_text_value

# %%
gt_text_value['segmentations'] = gazetteer._segment_companies(gt_text_value["text"].apply(lambda x: f" {x} ")).dropna()

# %%
gt_text_value

# %%
pd.set_option("max_rows", 10, "min_rows", 10)
gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
gazetteer.config["stop_words"] = False
gazetteer.config["stop_words_percent"] = 0
gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
gazetteer.prepare_to_segment()

gazetteer_df = pd.DataFrame(gazetteer.segmenter, columns=["company_name", "company_regex"])
print(len(gazetteer_df))
# gazetteer_df.to_html("extreme_untrained_gazetteer.html")


# %%
gt_seg_df = pd.DataFrame()
gt_seg_df["gt_text"] = df["PAST_CLIENT-gt_text"]
gt_seg_df["gt_text_joined"] = gt_seg_df["gt_text"].apply(lambda x: " ".join(x))

# %%
gt_seg_df = gt_seg_df[gt_seg_df["gt_text"].apply(len) > 0 ]

# %%
# gt_seg_df

# %%
optimal_paral = OptimalParallel()
company_spam = optimal_paral.parallelize_optimally(
    series=gt_seg_df["gt_text_joined"],
    series_measurement=gt_seg_df["gt_text_joined"].apply(len),
    function=gazetteer._segment_companies,
)

# %%
gt_seg_df["company_spam"] = company_spam

# %%
df["comany_spam"] = company_spam

# %%
df

# %%
gt_seg_df
