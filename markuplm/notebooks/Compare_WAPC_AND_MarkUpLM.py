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
#     display_name: wae
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare WAPC in production with a MarkUpLM

# %% tags=[]
from microcosm_sagemaker.bundle import BundleInputArtifact
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.common.utils.parallel_tools import OptimalParallel
from web_annotation_extractor.bundles.past_client.bundle import PastClientBundle
from web_annotation_extractor.bundles.past_client.segmentation import *
from web_annotation_extractor.evaluations.metric_functions import get_metrics_for_domain, calculate_metrics_for_dataset
from web_annotation_extractor.evaluations.visualization_functions import plot_performance

graph = create_object_graph('test')
pd.set_option('max_columns',60, 'max_colwidth',80, 'max_rows',5)

# %% [markdown]
# ## Load the SWDE-CF data

# %% tags=[]
dataset = 'develop'
print(dataset)
if dataset == 'develop':
    data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1735)_neg(4032)_intermediate.pkl"
df = pd.read_pickle(data_path)

print(len(df))
df = df.set_index("url")
df.head(2)

# %%
df["html"] = df["html"].str.replace("&amp;", "&")
df["html"] = df["html"].str.replace("&AMP;", "&")

# %%
df['domain'].value_counts()

# %%
tag="PAST_CLIENT"

# %% [markdown]
# ## Load and Predict WAPC model

# %% tags=[]
pc = PastClientBundle(graph)
pc.load(BundleInputArtifact("../../../globality-ml-scripts/wae_models/model/past_client_bundle/"))

# %% tags=[]
df['extracted_entities'] = pc.predict_batch(df.index, df['html'], df['content_type'])

# %% tags=[]
df['extracted_entities'].dropna().apply(len).value_counts()

# %% tags=[]
df[f'{tag}-gt_value'].dropna().apply(len).value_counts()

# %% [markdown]
# ### Get WAPC metrics

# %% tags=[]
df["predicted_value"] = df["extracted_entities"].apply(lambda extraction: [str(w) for z in [y.taxonomy_uris for y in extraction] for w in z])
# df['predicted_value_len'] = df['predicted_value'].apply(len)
# df = df.sort_values('predicted_value_len')
df["predicted_value_untax"] = df["extracted_entities"].apply(lambda extraction: [y.text for y in extraction]) #! If I change the way the extracted entities are generating `text`` I will have to change it here too

# %%
negative_percentage = 0.1

metrics_per_domain = {}
for domain_name, domain_df in df.groupby("domain"):
    metrics_per_domain[domain_name] = get_metrics_for_domain(
        ground_truth_series = domain_df[f"{tag}-gt_value"],
        prediction_series = domain_df["predicted_value"],
        annotations_series = domain_df["PAST_CLIENT-annotations"],
        negative_percentage = negative_percentage)
metrics_per_domain = pd.DataFrame(metrics_per_domain).T.sort_values("num_positives")
metrics_per_domain

# %%
metrics_per_domain.describe()[["precision","precision_adjusted","recall","f1"]].loc[["mean", "count"]]


# %% [markdown]
# ### Get new metrics and plots

# %%
def get_metrics_and_plots(
    df,
    gt_col=f"PAST_CLIENT-gt_value_untax",
    predicted_col=f"predicted_value_untax",
    annotations_col="PAST_CLIENT-annotations",
    negative_percentage=0.1,
):

    metrics_per_domain = dict()
    for domain_name, domain_df in df.groupby("domain"):
        metrics_per_domain[domain_name] = get_metrics_for_domain(
            ground_truth_series=domain_df[gt_col],
            prediction_series=domain_df[predicted_col],
            annotations_series=domain_df[annotations_col],
            negative_percentage=negative_percentage,
        )
        metrics_per_domain[domain_name][gt_col] = sorted(
            list(set([z for y in domain_df[gt_col] for z in y]))
        )
        metrics_per_domain[domain_name][predicted_col] = sorted(
            list(set([z for y in domain_df[predicted_col] for z in y]))
        )

    metrics_per_domain = pd.DataFrame(metrics_per_domain).T.sort_values("num_positives")
    len(metrics_per_domain)

    # # ? Log metrics per domain table
    # metrics_per_domain

    # # ? There are some domains with gt_text but not with gt_value
    metrics_per_domain = metrics_per_domain[metrics_per_domain["num_positives"] > 0]
    len(metrics_per_domain)

    # # ? Log metrics per domain table
    # metrics_per_domain

    fig = plot_performance(
        title="WAPC Performance per Domain ",
        domain=metrics_per_domain.index,
        TP=metrics_per_domain["TP"],
        FP=metrics_per_domain["FP"],
        FN=metrics_per_domain["FN"],
        num_positives=metrics_per_domain["num_positives"],
        precision=metrics_per_domain["precision"],
        recall=metrics_per_domain["recall"],
        f1=metrics_per_domain["f1"],
    )
    fig_adjusted = plot_performance(
        title="WAPC Performance per Domain Adjusted",
        domain=metrics_per_domain.index,
        TP=metrics_per_domain["TP"],
        FP=metrics_per_domain["FP_adjusted"],
        FN=metrics_per_domain["FN"],
        num_positives=metrics_per_domain["num_positives"],
        precision=metrics_per_domain["precision_adjusted"],
        recall=metrics_per_domain["recall"],
        f1=metrics_per_domain["f1"],
    )

    # # ? Log figures
    save_path = f"metrics_and_plots/plot_perf_domain.html"
    fig.write_html(save_path)
    save_path = f"metrics_and_plots/plot_perf_domain_adjusted.html"
    fig_adjusted.write_html(save_path)

    # # ? Compute dataset metrics
    metrics_dataset = calculate_metrics_for_dataset(metrics_per_domain)

    # # ? Log dataset metrics
    return metrics_dataset, metrics_per_domain, fig, fig_adjusted


# %%

# %%
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
    df,
    gt_col="PAST_CLIENT-gt_value_untax",
    predicted_col="predicted_value_untax",
    annotations_col="PAST_CLIENT-annotations",
    negative_percentage=negative_percentage,
)


# %%
metrics_dataset

# %%
metrics_per_domain

# %%
metrics_per_domain[["precision","precision_adjusted","recall","f1"]].fillna(pd.NA).dropna()

# %%
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
    df,
    gt_col="PAST_CLIENT-gt_value_untax",
    predicted_col="predicted_value_untax",
    annotations_col="PAST_CLIENT-annotations",
    negative_percentage=negative_percentage,
)


# %%
metrics_dataset

# %%
metrics_dataset

# %%
metrics_per_domain

# %% [markdown]
# # Load Results from MarkupLM Trained Model and display metrics 

# %% [markdown]
# ---

# %%
segment_all_nodes = False
trained = False

if segment_all_nodes:
    results = pd.read_pickle("results_classified_5_epoch.pkl")
    results = results.reset_index().drop('index', axis=1)
    results["text"] = results["text"].apply(lambda x: f" {x} ")

    results["text"] = results["text"].str.replace("&amp;", "&")
    results["text"] = results["text"].str.replace("&AMP;", "&")
    if trained:
        save_path = "results_classified_5_epoch_segmented_trained.pkl"
    else:
        save_path = "results_classified_5_epoch_segmented_untrained.pkl"
else:
    if trained:
        data_path = "results_classified_5_epoch_segmented_trained.pkl"
    else:
        data_path = "results_classified_5_epoch_segmented_untrained.pkl"
    results = pd.read_pickle(data_path)

# %%
# # #? Segment all nodes data
# segment_all_nodes = False
# if segment_all_nodes:
#     optimal_paral = OptimalParallel()
#     node_company_spam = optimal_paral.parallelize_optimally(
#         series=results["text"],
#         series_measurement=results["text"].apply(len),
#         function=pc.segmenter_html._segment_companies,
#     )
#     node_company_spam = node_company_spam.fillna("").apply(list)

#     results["node_company_spam"] = node_company_spam
#     results["node_companies"] = node_company_spam.apply(
#         lambda company_spam: [x[0] for x in company_spam]
#     )
#     results["node_spams"] = node_company_spam.apply(
#         lambda company_spam: [x[0] for x in company_spam]
#     )

#     #? Create mapping to convert value into taxonomy
#     value_to_taxonomy_mappings = dict([(company.name, company.uri)for company in graph.known_company_taxonomy])

#     #? Convert reconciliations into reconciliations taxonomized 
#     results['node_companies_tax'] = results['node_companies'].dropna().apply(lambda values: [value_to_taxonomy_mappings.get(x) for x in values]) 
#     results['node_companies_tax'] = results['node_companies_tax'].fillna("").apply(list)

#     results.to_pickle("results_classified_5_epoch_segmented.pkl")

#     # results['node_reconciliations'] = pc.get_reconciliations(results['node_segmentations'])

# %%
# # ? Segment all nodes data
if segment_all_nodes:
    if trained:
        gazetteer = pc.segmenter_html._segment_companies
    else:
        from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
        gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
        gazetteer.config["stop_words"] = True
        gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
        gazetteer.prepare_to_segment()

    print(len(gazetteer.segmenter))
    
    optimal_paral = OptimalParallel()
    node_company_spam = optimal_paral.parallelize_optimally(
        series=results["text"],
        series_measurement=results["text"].apply(len),
        function=gazetteer._segment_companies,
    )

    node_company_spam = node_company_spam.fillna("").apply(list)

    results["node_company_spam"] = node_company_spam
    results["node_companies"] = node_company_spam.apply(
        lambda company_spam: [x[0] for x in company_spam]
    )
    results["node_spams"] = node_company_spam.apply(
        lambda company_spam: [x[0] for x in company_spam]
    )

    # # ? Create mapping to convert value into taxonomy
    value_to_taxonomy_mappings = dict(
        [(company.name, company.uri) for company in graph.known_company_taxonomy]
    )

    # # ? Convert reconciliations into reconciliations taxonomized
    results["node_companies_tax"] = (
        results["node_companies"]
        .dropna()
        .apply(lambda values: [value_to_taxonomy_mappings.get(x) for x in values])
    )
    results["node_companies_tax"] = results["node_companies_tax"].fillna("").apply(list)

    results.to_pickle(save_path)

# %%
mode_indices = dict(
    model=results[results["pred_type"] == "PAST_CLIENT"].index,
    ground_truth=results[results["truth"] == "PAST_CLIENT"].index,
    no_classification=results.index,
)

for mode, index in mode_indices.items():
    print(mode, len(index))
    results[f"{mode}-node_companies_tax"] = results["node_companies_tax"]
    results.loc[~results.index.isin(index), f"{mode}-node_companies_tax"] = ''
    results[f"{mode}-node_companies_tax"] = results[f"{mode}-node_companies_tax"].apply(list)

# %%
model_count = results['model-node_companies_tax'].apply(len).sum()
ground_truth_count = results['ground_truth-node_companies_tax'].apply(len).sum()
no_classification_count = results['no_classification-node_companies_tax'].apply(len).sum()
model_count, ground_truth_count, no_classification_count

# %%
# # ? Group reconciliations per node into reconcilaiton per page
results_grouped = pd.DataFrame(
    results.groupby(by="html_path").agg(
        {
            "node_companies_tax": lambda x: [z for y in list(x) for z in y],
            "model-node_companies_tax": lambda x: [z for y in list(x) for z in y],
            "ground_truth-node_companies_tax": lambda x: [z for y in list(x) for z in y],
            "no_classification-node_companies_tax": lambda x: [z for y in list(x) for z in y],
            "domain": lambda x: list(x)[0],
        }
    )
)

# # ? Load and apply pageid to url mapping
pageid_url_mapping = pd.read_pickle(
    "/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl"
)
results_grouped.reset_index(inplace=True)
results_grouped["url"] = results_grouped["html_path"].apply(
    lambda x: pageid_url_mapping.get(x)[0]
)
results_grouped = results_grouped.drop(["domain"], axis=1)

# # ? Set index from both dataframes
results_grouped = results_grouped.set_index("url")

# %% [markdown]
# ### Merge develop with results_grouped (predictions from MarkupLM)

# %%
merge = df.join(results_grouped).reset_index()

merge["node_companies_tax"] = merge["node_companies_tax"].fillna("").apply(list)
for mode, index in mode_indices.items():
    merge[f"{mode}-node_companies_tax"] = merge[f"{mode}-node_companies_tax"].fillna("").apply(list)

# %% [markdown]
# ### Compute Metrics

# %%
print("WAPC")
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
    merge,
    gt_col="PAST_CLIENT-gt_value_untax",
    predicted_col="predicted_value_untax",
    annotations_col="PAST_CLIENT-annotations",
)
display(metrics_dataset)

for mode in mode_indices:
    print(mode)
    metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
        merge,
        gt_col="PAST_CLIENT-gt_value",
        predicted_col=f"{mode}-node_companies_tax",
        annotations_col="PAST_CLIENT-annotations",
    )
    display(metrics_dataset)

# %%
print("WAPC")
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
    merge,
    gt_col="PAST_CLIENT-gt_value_untax",
    predicted_col="predicted_value_untax",
    annotations_col="PAST_CLIENT-annotations",
)
display(metrics_dataset)

for mode in mode_indices:
    print(mode)
    metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(
        merge,
        gt_col="PAST_CLIENT-gt_value",
        predicted_col=f"{mode}-node_companies_tax",
        annotations_col="PAST_CLIENT-annotations",
    )
    display(metrics_dataset)

# %%
print("WAPC")
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, gt_col="PAST_CLIENT-gt_value_untax", predicted_col="predicted_value_untax")
display(metrics_dataset)

for mode in mode_indices:
    print(mode)
    metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col=f"{mode}-node_companies_tax", gt_col="PAST_CLIENT-gt_value")
    display(metrics_dataset)

# %% [markdown]
# ---

# %%
results[results["domain"] == "palisade.com.pickle"].sort_values("PAST_CLIENT-node_gt_text")

# %%
with pd.option_context("max_rows", 200, "min_rows", 200):
    display(results[results["domain"] == "palisade.com.pickle"].sort_values("html_path"))

# %%
len(gazetteer.segmenter)

# %%
# #? Using the text from the values, get the maximum performance (this helps to identify the maximum recall the can give)
merge_text_value = merge[merge["PAST_CLIENT-gt_value"].apply(len)>0]
segmentations = gazetteer._segment_companies(merge_text_value["PAST_CLIENT-gt_text"].apply(lambda x: ' ' + ' , '.join(x) + ' ')).dropna()
merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[0]) for x in y])
merge["text_value_segmentations"] = merge["text_value_segmentations"].fillna('').apply(list)

merge_text_value.dropna()
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col="text_value_segmentations", gt_col="PAST_CLIENT-gt_value", annotations_col="PAST_CLIENT-annotations")
display(metrics_dataset)

# %%
# #? Using the text from the values, get the maximum performance (this helps to identify the maximum recall the can give)
merge_text_value = merge[merge["PAST_CLIENT-gt_value"].apply(len)>0]
segmentations = gazetteer._segment_companies(merge_text_value["PAST_CLIENT-gt_text"].apply(lambda x: ' ' + ' , '.join(x) + ' ')).dropna()
# merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[0]) for x in y])
merge["text_value_segmentations"] = segmentations.apply(lambda y: [x[0] for x in y])
merge["text_value_segmentations"] = merge["text_value_segmentations"].fillna('').apply(list)

merge_text_value.dropna()
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col="text_value_segmentations", gt_col="PAST_CLIENT-gt_value_untax", annotations_col="PAST_CLIENT-annotations")
display(metrics_dataset)

# %%
# #? Using the text from the values, get the maximum performance (this helps to identify the maximum recall the can give)
merge_text_value = merge[merge["PAST_CLIENT-gt_value"].apply(len)>0]
segmentations = gazetteer._segment_companies(merge_text_value["PAST_CLIENT-gt_text"].apply(lambda x: ' ' + ' , '.join(x) + ' ')).dropna()
# merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[0]) for x in y])
merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[1]) for x in y])
merge["text_value_segmentations"] = merge["text_value_segmentations"].fillna('').apply(list)

merge_text_value.dropna()
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col="text_value_segmentations", gt_col="PAST_CLIENT-gt_value_untax", annotations_col="PAST_CLIENT-annotations")
display(metrics_dataset)

# %%
merge_text_value.loc[1734]["url"]

# %%
# merge
with pd.option_context("max_colwidth", 3000):
    display(merge[["text_value_segmentations", "PAST_CLIENT-gt_value_untax", "PAST_CLIENT-gt_text"]])


# %%
l = [x[0] for x in gazetteer.segmenter]
[y for y in l if "7-Eleven" in y]

# %%

# %%
pd.DataFrame(df.loc["https://www.walkerhamill.com/credentials/all"]).T["source_batch"]

# %%
metrics_per_domain

# %%
metrics_per_domain_html = metrics_per_domain.to_html()

# %%
with open("metrics_per_domain_html.html", "w") as fs:
    fs.write(metrics_per_domain_html)

# %%
from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
gazetteer.config["stop_words"] = True
gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
gazetteer.prepare_to_segment()

value_to_taxonomy_mappings = dict(
        [(company.name, company.uri) for company in graph.known_company_taxonomy]
    )

# %%
len(pc.segmenter_html.segmenter)

# %%
# #? Using the text from the values, get the maximum performance (this helps to identify the maximum recall the can give)
merge_text_value = merge[merge["PAST_CLIENT-gt_value"].apply(len)>0]
segmentations = pc.segmenter_html._segment_companies(merge_text_value["PAST_CLIENT-gt_text"].apply(lambda x: ' ' + ' , '.join(x) + ' ')).dropna()
merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[0]) for x in y])
merge["text_value_segmentations"] = merge["text_value_segmentations"].fillna('').apply(list)

merge_text_value.dropna()
metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col="text_value_segmentations", gt_col="PAST_CLIENT-gt_value")
display(metrics_dataset)

# %%
metrics_per_domain2 = metrics_per_domain

# %%
# Create a value counts of gt_text [Done]
# Create a value counts of gt_value [Done]
# How many gt_text are written exactly like the name of our company?
# Create a function that shows the FP, TP, FN of the segmenter per gt_text 
# Create visualization of gt_value | gt_text

gt_text_and_gt_value = pd.DataFrame([(x["gt_value_untax"], x["gt_text"]) for l in df["PAST_CLIENT-annotations-untax"] for x in l ], columns=["gt_value", "gt_text"]).sort_values("gt_value")
gt_text_and_gt_value_drop = gt_text_and_gt_value.dropna().drop_duplicates("gt_text").reset_index().drop("index",axis=1)
with open("gt_text_and_gt_value.html", "w") as fs:
    fs.write(gt_text_and_gt_value_drop.to_html())

# %%
distinct_companies_in_taxonomy = gazetteer.companies
distinct_companies_in_dataset = sorted(list(set(gt_text_and_gt_value["gt_value"].dropna().str.lower())))
print(f"Number of distinct companies in:\n dataset = {len(distinct_companies_in_dataset)}\n taxonomy = {len(distinct_companies_in_taxonomy)} \n ({100*len(distinct_companies_in_dataset) / len(distinct_companies_in_taxonomy):.1f} %)")


# %%
print("Number of companies in gt_value that their name are equal to gt_text - That will tell us how much can we rely on a perfect name string matching.")
(gt_text_and_gt_value_drop["gt_value"].str.lower() == gt_text_and_gt_value_drop["gt_text"].str.lower()).value_counts()

# %%
print("Number of times a gt_text company appears in dataset:")
with pd.option_context("max_rows", 20, "min_rows", 20):
    display(gt_text_and_gt_value["gt_text"].str.lower().value_counts())


# %%
print("Number of times a gt_value company appears in dataset:")
with pd.option_context("max_rows", 20, "min_rows", 20):
    display(gt_text_and_gt_value["gt_value"].value_counts())
