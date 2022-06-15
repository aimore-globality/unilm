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

# %% [markdown]
# # Compare WAPC in production with a MarkUpLM

# %% tags=[]
from microcosm_sagemaker.bundle import BundleInputArtifact
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.common.utils.parallel_tools import OptimalParallel
from web_annotation_extractor.bundles.past_client.bundle import PastClientBundle
from web_annotation_extractor.bundles.past_client.segmentation import *
from web_annotation_extractor.evaluations.metric_functions import get_reconciliations_metrics_for_all_domains, average_domains_metrics, calculate_metrics_for_dataset, combine_and_get_set, combine_and_get_sorted_list
from web_annotation_extractor.evaluations.visualization_functions import plot_performance
from pathlib import Path
graph = create_object_graph('test')
pd.set_option('max_columns',60, 'max_colwidth',80, 'max_rows',5)
import wandb

run = wandb.init(project="LanguageModel", resume=False, tags=["compare_with_production"])


# %%
segmenter_trained =["trained", "untrained", "extreme_untrained"][0]
predict_and_segment = True

tag="PAST_CLIENT"
negative_percentage = 0.1

# %% [markdown]
# ## Load the SWDE-CF data

# %% tags=[]
save_load_data_path = Path(f"develop_WAPC_cache/develop_{segmenter_trained}_WAPC.pkl")

if predict_and_segment:
    if not save_load_data_path.exists():
        save_load_data_path.mkdir(parents=True, exist_ok=True)
    print(f"save_load_data_path: {save_load_data_path}")
    
    # # ? Load Data
    dataset = 'develop'
    print(dataset)
    if dataset == 'develop':
        # data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1765)_neg(4083)_intermediate.pkl"
        data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1765)_neg(4086)_intermediate.pkl"
        
    df = pd.read_pickle(data_path)

    print(len(df))
    df = df.set_index("url")
    df.head(2)

    df["html"] = df["html"].str.replace("&amp;", "&")
    df["html"] = df["html"].str.replace("&AMP;", "&")

    df['domain'].value_counts()

    # # ? Load Model
    pc = PastClientBundle(graph)
    pc.load(BundleInputArtifact("../../../globality-ml-scripts/wae_models/model/past_client_bundle/"))

    if segmenter_trained in ["untrained", "extreme_untrained"]:
        from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
        gazetteer_html = PastClientSegmenter(graph.html_gazetteer.config)
        gazetteer_url = PastClientSegmenter(graph.url_gazetteer.config)

        gazetteer_html.config["stop_words"] = True
        gazetteer_url.config["stop_words"] = True

        if segmenter_trained == "extreme_untrained":
            gazetteer_html.config["stop_words_percent"] = 0
            gazetteer_url.config["stop_words_percent"] = 0

        gazetteer_html.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
        gazetteer_html.prepare_to_segment()
        
        gazetteer_url.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
        gazetteer_url.prepare_to_segment()

        pc.segmenter_html = gazetteer_html
        pc.segmenter_url = gazetteer_url

    print(f"segmenter_html: {len(pc.segmenter_html.segmenter)}")
    print(f"segmenter_url: {len(pc.segmenter_url.segmenter)}")

    # # ? Predict
    df['extracted_entities'] = pc.predict_batch(df.index, df['html'], df['content_type'])

    df["WAPC-node_company_span_taxonomy"] = df["extracted_entities"].apply(lambda extraction: [(y.text, y.text, y.taxonomy_uris[0]) for y in extraction]) #! If I change the way the extracted entities are generating `text`` I will have to change it here too
    
    df.to_pickle(save_load_data_path)
else:
    df = pd.read_pickle(save_load_data_path)


# %% [markdown]
# ### Get WAPC metrics

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(
    df = df,
    gt_col = f"{tag}-gt_value",
    predicted_col = "WAPC-node_company_span_taxonomy",
    annotations_col = "PAST_CLIENT-annotations",
    negative_percentage = negative_percentage)
average_domains_metrics(domain_metrics)

# %% [markdown]
# ### Get new metrics and plots

# %%
# def get_metrics_and_plots(
#     df,
#     gt_col=f"PAST_CLIENT-gt_value_untax",
#     predicted_col=f"predicted_value_untax",
#     annotations_col="PAST_CLIENT-annotations",
#     negative_percentage=0.1,
# ):

#     metrics_per_domain = dict()
#     for domain_name, domain_df in df.groupby("domain"):
#         metrics_per_domain[domain_name] = get_metrics_for_domain(
#             ground_truth_series=domain_df[gt_col],
#             prediction_series=domain_df[predicted_col],
#             annotations_series=domain_df[annotations_col],
#             negative_percentage=negative_percentage,
#         )
#         metrics_per_domain[domain_name][gt_col] = sorted(
#             list(set([z for y in domain_df[gt_col] for z in y]))
#         )
#         metrics_per_domain[domain_name][predicted_col] = sorted(
#             list(set([z for y in domain_df[predicted_col] for z in y]))
#         )

#     metrics_per_domain = pd.DataFrame(metrics_per_domain).T.sort_values("num_positives")
#     len(metrics_per_domain)

#     # ? Log metrics per domain table
#     # metrics_per_domain

#     # ? There are some domains with gt_text but not with gt_value
#     metrics_per_domain = metrics_per_domain[metrics_per_domain["num_positives"] > 0]
#     len(metrics_per_domain)

#     # ? Log metrics per domain table
#     # metrics_per_domain

#     fig = plot_performance(
#         title="WAPC Performance per Domain ",
#         domain=metrics_per_domain.index,
#         TP=metrics_per_domain["TP"],
#         FP=metrics_per_domain["FP"],
#         FN=metrics_per_domain["FN"],
#         num_positives=metrics_per_domain["num_positives"],
#         precision=metrics_per_domain["precision"],
#         recall=metrics_per_domain["recall"],
#         f1=metrics_per_domain["f1"],
#     )
#     fig_adjusted = plot_performance(
#         title="WAPC Performance per Domain Adjusted",
#         domain=metrics_per_domain.index,
#         TP=metrics_per_domain["TP"],
#         FP=metrics_per_domain["FP_adjusted"],
#         FN=metrics_per_domain["FN"],
#         num_positives=metrics_per_domain["num_positives"],
#         precision=metrics_per_domain["precision_adjusted"],
#         recall=metrics_per_domain["recall"],
#         f1=metrics_per_domain["f1"],
#     )

#     # ? Log figures
#     save_path = f"metrics_and_plots/plot_perf_domain.html"
#     fig.write_html(save_path)
#     save_path = f"metrics_and_plots/plot_perf_domain_adjusted.html"
#     fig_adjusted.write_html(save_path)

#     # ? Compute dataset metrics
#     metrics_dataset = calculate_metrics_for_dataset(metrics_per_domain)

#     # ? Log dataset metrics
#     return metrics_dataset, metrics_per_domain, fig, fig_adjusted

# %% [markdown]
# # Load Results from MarkupLM Trained Model and display metrics 

# %%
# classified_nodes_folder_root = Path("results_classified")

# if not classified_nodes_folder_root.exists():
#     classified_nodes_folder_root.mkdir(parents=True, exist_ok=True)

# # #? Put the predictions of the model inside the folder above and select the file with `classified_nodes_data_path`
# # classified_nodes_data_path = "results_classified_5_epoch.pkl"
# # classified_nodes_data_path = "develop_set_nodes_classified_epoch_10.pkl"
# classified_nodes_data_path = "develop_set_nodes_classified_epoch_4_dedup.pkl"
# # classified_nodes_data_path = "develop_set_nodes_classified_epoch_4.pkl"



# save_load_data_path = f"{classified_nodes_data_path.split('.pkl')[0]}_segmented_{segmenter_trained}.pkl"
# print(f"save_load_data_path: {save_load_data_path}")

# if predict_and_segment:
#     load_file = str(classified_nodes_folder_root / classified_nodes_data_path)
#     print(f"Load file: {load_file}")
#     results = pd.read_pickle(load_file)
#     display(results.head())
#     results = results.reset_index().drop('index', axis=1)
#     results["text"] = results["text"].apply(lambda x: f" {x} ")

#     results["text"] = results["text"].str.replace("&amp;", "&")
#     results["text"] = results["text"].str.replace("&AMP;", "&")

#     if segmenter_trained in ["trained"]:
#         gazetteer = pc.segmenter_url # TODO: Replace for the segmenter_html
#     else:
#         from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
#         gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
#         gazetteer.config["stop_words"] = True
#         if segmenter_trained == "extreme_untrained":
#             gazetteer.config["stop_words_percent"] = 0
#         gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
#         gazetteer.prepare_to_segment()

#     print(len(gazetteer.segmenter))
#     optimal_paral = OptimalParallel()
#     node_company_span = optimal_paral.parallelize_optimally(
#         series=results["text"],
#         series_measurement=results["text"].apply(len),
#         function=gazetteer._segment_companies,
#     )

#     node_company_span = node_company_span.fillna("").apply(list)

#     value_to_taxonomy_mappings = dict([(company.name, company.uri) for company in graph.known_company_taxonomy])

#     results["node_company_span_taxonomy"] = node_company_span.apply(lambda company_span: [(x[0], x[1], value_to_taxonomy_mappings.get(x[0])) for x in company_span])
    
#     results.to_pickle(save_load_data_path)
# else:
#     results = pd.read_pickle(save_load_data_path)

# %%
from microcosm.api import create_object_graph
import pandas as pd
from pathlib import Path

# %%
results

# %%
taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])

# Train Segmenter on gt nodes
classified_nodes_folder_root = Path("results_classified")

if not classified_nodes_folder_root.exists():
    classified_nodes_folder_root.mkdir(parents=True, exist_ok=True)

# #? Put the predictions of the model inside the folder above and select the file with `classified_nodes_data_path`
# classified_nodes_data_path = "results_classified_5_epoch.pkl"
# classified_nodes_data_path = "develop_set_nodes_classified_epoch_10.pkl"
# classified_nodes_data_path = "develop_set_nodes_classified_epoch_4_dedup.pkl"
# classified_nodes_data_path = "train_set_nodes_classified_epoch_4_dedup.pkl"
# classified_nodes_data_path = "develop_set_nodes_classified_epoch_4_dedup.pkl"
# classified_nodes_data_path = "develop_set_nodes_classified_epoch_4.pkl"
classified_nodes_data_path = "/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/develop_df_pred.pkl"


load_file = str(classified_nodes_folder_root / classified_nodes_data_path)
print(f"Load file: {load_file}")
results = pd.read_pickle(load_file)

convert_text_into_segmentation_value_map = {a['text']:a['value'] for a in [z for y in [x for x in df["annotations"].apply(lambda x: x.get("PAST_CLIENT")).values if x] for z in y] if a['value']}
results["value"] = results.apply(lambda row: [convert_text_into_segmentation_value_map.get(x) for x in row[f"{tag}-gt_text"]], axis=1)
results["value"] = results["value"].apply(lambda row: [taxonomy_to_value_mappings.get(x) for x in row])

results = results.reset_index().drop('index', axis=1)
results["node_text"] = results["node_text"].apply(lambda x: f" {x} ")

results["node_text"] = results["node_text"].str.replace("&amp;", "&")
results["node_text"] = results["node_text"].str.replace("&AMP;", "&")

# #? Train a new gazetter:
# from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
# gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
# gazetteer.config["stop_words"] = True
# gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
# gazetteer.prepare_to_segment()

# data_for_gazetteer_to_train = results[results["truth"] != 'none']

# gazetteer.input_type = "text"

# data_for_gazetteer_to_train['text_len'] = data_for_gazetteer_to_train['text'].apply(len)
# data_for_gazetteer_to_train = gazetteer.segment(data_for_gazetteer_to_train)
# gazetteer.fit(data_for_gazetteer_to_train)

# #? Load trained gazetter:
gazetteer = pd.read_pickle("trained_gazetteer.pkl")

print(len(gazetteer.segmenter))
optimal_paral = OptimalParallel()
node_company_span = optimal_paral.parallelize_optimally(
    series=results["node_text"],
    series_measurement=results["node_text"].apply(len),
    function=gazetteer._segment_companies,
)

node_company_span = node_company_span.fillna("").apply(list)

value_to_taxonomy_mappings = dict([(company.name, company.uri) for company in graph.known_company_taxonomy])

results["node_company_span_taxonomy"] = node_company_span.apply(lambda company_span: [(x[0], x[1], value_to_taxonomy_mappings.get(x[0])) for x in company_span])

# %%
# # ? Get the Segmentations (company_span_taxonomy)
mode_indices = dict(
    model=results[results["node_pred_tag"] == "PAST_CLIENT"].index,
    ground_truth=results[results["node_gt_tag"] == "PAST_CLIENT"].index,
    no_classification=results.index,
)
print("# Nodes:")
for mode, index in mode_indices.items():
    print(f"{mode}: {len(index)}")
    results[f"{mode}-node_company_span_taxonomy"] = results["node_company_span_taxonomy"]
    results.loc[~results.index.isin(index), f"{mode}-node_company_span_taxonomy"] = ''
    results[f"{mode}-node_company_span_taxonomy"] = results[f"{mode}-node_company_span_taxonomy"].apply(list)

model_count = results['model-node_company_span_taxonomy'].apply(len).sum()
ground_truth_count = results['ground_truth-node_company_span_taxonomy'].apply(len).sum()
no_classification_count = results['no_classification-node_company_span_taxonomy'].apply(len).sum()
print(f"# Companies Found: \nmodel_count: {model_count}, ground_truth_count: {ground_truth_count}, no_classification_count: {no_classification_count}")

# %%
# results["domain"] = results["html_path"].apply(lambda x: x.split(".pickle-0000.htm")[0])

# %%
results["node_gt_tag"].value_counts()

# %%
results

# %%
threshold = 0.5
results["node_gt_tag"] = results["node_prob"].apply(lambda x: "PAST_CLIENT" if x > threshold else 'none')
results_capped = results.copy(deep=True)

columns = [
"model-node_company_span_taxonomy",
]

for c in columns:
    results_capped[c] = results_capped[c].mask(results_capped["node_pred_tag"] == 'none', '')
    results_capped[c] = results_capped[c].apply(list)
    results_capped[c].value_counts()

# %%
results_capped["node_gt_tag"].value_counts()

# %%
# # ? Group reconciliations per node into reconciliation per page
results_grouped = pd.DataFrame(
    results_capped.groupby(by="html_path").agg(
        {
            "node_company_span_taxonomy": lambda x: combine_and_get_set(x), 
            "model-node_company_span_taxonomy": lambda x: combine_and_get_set(x),
            "ground_truth-node_company_span_taxonomy": lambda x: combine_and_get_set(x),
            "no_classification-node_company_span_taxonomy": lambda x: combine_and_get_set(x),
            "domain": lambda x: list(x)[0],
        }
    )
)

# # ? Load and apply pageid to url mapping
# pageid_url_mapping = pd.read_pickle(
#     "/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl"
# )
results_grouped.reset_index(inplace=True)
# results_grouped["url"] = results_grouped["html_path"].apply(
#     lambda x: pageid_url_mapping.get(x)[0]
# )
results_grouped = results_grouped.drop(["domain"], axis=1)

# # ? Set index from both dataframes
results_grouped = results_grouped.set_index("url")

# %% [markdown]
# ### Merge develop with results_grouped (predictions from MarkupLM)

# %%
merge = df.join(results_grouped).reset_index()

merge["node_company_span_taxonomy"] = merge["node_company_span_taxonomy"].fillna("").apply(list)
for mode, index in mode_indices.items():
    merge[f"{mode}-node_company_span_taxonomy"] = merge[f"{mode}-node_company_span_taxonomy"].fillna("").apply(list)

# %% [markdown]
# ### Compute Metrics

# %%
from pathlib import Path
print(f"threshold: {threshold}")
taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+['model']:
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
from pathlib import Path
print(f"threshold: {threshold}")
taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+['model']:
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
from pathlib import Path

taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])
print(f"threshold: {threshold}")

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+list(mode_indices.keys()):
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
from pathlib import Path

taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])
print(f"threshold: {threshold}")

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+list(mode_indices.keys()):
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
from pathlib import Path

taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+list(mode_indices.keys()):
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
from pathlib import Path

taxonomy_to_value_mappings = dict([(company.uri, company.name) for company in graph.known_company_taxonomy])

print(f"Metrics with the Segmenter {segmenter_trained}!")

for mode in ["WAPC"]+list(mode_indices.keys()):
    print(mode)
    domain_metrics = get_reconciliations_metrics_for_all_domains(
        df=merge,
        gt_col=f"{tag}-gt_value",
        predicted_col=f"{mode}-node_company_span_taxonomy",
        annotations_col="PAST_CLIENT-annotations",
        negative_percentage=negative_percentage,
    )
    display(calculate_metrics_for_dataset(domain_metrics))

    folder_path = Path(f"/data/GIT/unilm/markuplm/notebooks/Analysys_Gazetteer/segmenter_{segmenter_trained}")
    folder_path.mkdir(parents=True, exist_ok=True)
    folder_path = str(folder_path)

    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["TP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-TP_seg.html")
    pd.DataFrame(pd.Series(combine_and_get_sorted_list(domain_metrics["FP_seg"])).value_counts()).to_html(f"{folder_path}/{mode}-FP_seg.html")

    fn = pd.DataFrame(domain_metrics["FN_seg"])
    fn = fn["FN_seg"].explode().dropna()
    fn_df = pd.DataFrame()
    fn_df["FN_gt_value"] = fn.apply(lambda x: x[0])
    fn_df["FN_gt_text"] = fn.apply(lambda x: x[1])
    fn_df = pd.DataFrame(fn_df.groupby("FN_gt_value").aggregate('FN_gt_text'))
    fn_df.columns = ["FN_gt_value", "FN_gt_text_list"]
    fn_df["FN_gt_text_list"] = fn_df["FN_gt_text_list"].apply(list)
    fn_df["FN_gt_text_len"] = fn_df["FN_gt_text_list"].apply(len)
    fn_df = fn_df.sort_values("FN_gt_text_len", ascending=False)
    
    fn_df.to_html(f"{folder_path}/{mode}-FN_pred.html")
print(folder_path)

# %%
df[df["domain"] == "misoportal.com"]["PAST_CLIENT-gt_value_untax"].value_counts()
# df[df["domain"] == "datameer.com"]["PAST_CLIENT-gt_value_untax"].value_counts()

# %%
# [y for x in df[df['domain'].isin(['palisade.com'])]["PAST_CLIENT-gt_text"].values for y in x]
urls_with_many_gt_text = df.sort_values("PAST_CLIENT-gt_text_len", ascending=False).iloc[:20]
urls_with_many_gt_text[["domain", "PAST_CLIENT-gt_text_len"]].sort_values("domain")

# %%
df[df["PAST_CLIENT-gt_text_len"].between(1,5)].sort_values("PAST_CLIENT-gt_text_len")["PAST_CLIENT-gt_text_len"].sum()

# %%
[x for x in domain_metrics.loc['palisade.com']["PAST_CLIENT-gt_value"]][0:]

# %%

# %%
with pd.option_context('display.max_rows', 20, 'display.min_rows', 20):
    display(domain_metrics.sort_values("precision", ascending=True))

# %%
domain_metrics = get_reconciliations_metrics_for_all_domains(
    df=merge,
    gt_col=f"{tag}-gt_value",
    predicted_col=f"{mode}-node_company_span_taxonomy",
    annotations_col="PAST_CLIENT-annotations",
    negative_percentage=negative_percentage,
)

# %%
# with pd.option_context('min_rows', 200, 'max_rows', 200, 'max_colwidth', 200): 
#    display(pd.DataFrame(pd.Series([x for y in domain_metrics.FP_pred for x in y]).value_counts()))

# %%
# domain_metrics.drop(["TP_pred","FP_pred","FN_pred","TP_seg","FP_seg","FN_seg"], axis=1)

# %%
# with pd.option_context("max_rows", 200, "min_rows", 200):
#     display(results[results["domain"] == "palisade.com.pickle"].sort_values("html_path"))

# %%
# # #? Using the text from the values, get the maximum performance (this helps to identify the maximum recall the can give)
# merge_text_value = merge[merge["PAST_CLIENT-gt_value"].apply(len)>0]
# segmentations = gazetteer._segment_companies(merge_text_value["PAST_CLIENT-gt_text"].apply(lambda x: ' ' + ' , '.join(x) + ' ')).dropna()
# merge["text_value_segmentations"] = segmentations.apply(lambda y: [value_to_taxonomy_mappings.get(x[0]) for x in y])
# merge["text_value_segmentations"] = merge["text_value_segmentations"].fillna('').apply(list)

# merge_text_value.dropna()
# metrics_dataset, metrics_per_domain, fig, fig_adjusted = get_metrics_and_plots(merge, predicted_col="text_value_segmentations", gt_col="PAST_CLIENT-gt_value", annotations_col="PAST_CLIENT-annotations")
# display(metrics_dataset)

# %%
# from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
# gazetteer = PastClientSegmenter(graph.html_gazetteer.config)
# gazetteer.config["stop_words"] = True
# gazetteer.stop_word_path = "/data/GIT/web-annotation-extractor/data/processed/train/enwiki_vocab_word_freqs.csv"
# gazetteer.prepare_to_segment()

# value_to_taxonomy_mappings = dict(
#         [(company.name, company.uri) for company in graph.known_company_taxonomy]
#     )
# taxonomy_to_value_mappings = dict(
#         [(company.uri, company.name) for company in graph.known_company_taxonomy]
#     )

# %%
# print("Number of times a gt_value company appears in dataset:")
# with pd.option_context("max_rows", 20, "min_rows", 20):
#     display(gt_text_and_gt_value["gt_value"].value_counts())

# %%
run.save()
run.finish()
