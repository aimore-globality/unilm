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
# pd.set_option('max_columns',60, 'max_colwidth',80, 'max_rows',5)
import wandb

run = wandb.init(project="LanguageModel", resume=False, tags=["compare_with_production"])


# %%
segmenter_trained =["trained", "untrained", "extreme_untrained"][0]

tag="PAST_CLIENT"

# %% [markdown]
# ## Load the SWDE-CF data

# %% tags=[]
save_load_data_path = Path(f"develop_WAPC_cache/develop_{segmenter_trained}_WAPC.pkl")
predict_and_segment = True

if predict_and_segment:
    if not save_load_data_path.exists():
        save_load_data_path.mkdir(parents=True, exist_ok=True)
    print(f"save_load_data_path: {save_load_data_path}")
    
    # # ? Load Data
    dataset = 'develop'
    print(dataset)
    data_path = "/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1830)_neg(45273)_intermediate.pkl" # With the gt_images + full
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
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',300, 'display.min_rows',300)

df[f"WAPC-predictions"] = df[f"WAPC-node_company_span_taxonomy"].apply(lambda row: [x[2] for x in row if row])

# %%
from web_annotation_extractor.evaluations.exact_match_evaluation import ExactMatchEvaluation

domain_results = df[["WAPC-predictions", f"{tag}-gt_value", "domain"]].groupby('domain').agg({'WAPC-predictions': 'sum', "PAST_CLIENT-gt_value":'sum'})
precision = domain_results.apply(lambda x: ExactMatchEvaluation.compute_precision(set(x[f"{tag}-gt_value"]), set(x["WAPC-predictions"])), axis=1 ).mean()
recall = domain_results.apply(lambda x: ExactMatchEvaluation.compute_recall(set(x[f"{tag}-gt_value"]), set(x["WAPC-predictions"])), axis=1 ).mean()
f1 = domain_results.apply(lambda x: ExactMatchEvaluation.compute_f1(precision, recall))

print(f"Avg. Precision: {precision}\n   Avg. Recall: {recall}\n   Avg. F1: {f1}")

# %%
len(domain_results)

# %%
run.save()
run.finish()

# %%
df[[f"PAST_CLIENT-gt_value", "domain"]].groupby('domain').agg({"PAST_CLIENT-gt_value":'sum'})

# %%
# Production_metrics = domain_results.copy(deep=True)
Production_metrics = domain_results.copy(deep=True)
Production_metrics = Production_metrics.rename({"PAST_CLIENT-gt_value": "GroundTruth-Production", "WAPC-predictions": "Prediction-Production"}, axis=1)
Production_metrics["GroundTruth-Production"] = Production_metrics["GroundTruth-Production"].apply(set)
Production_metrics["Prediction-Production"] = Production_metrics["Prediction-Production"].apply(lambda row: sorted(list(set(row))))
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

Production_metrics

# %%
# folder = "model_results"
# domain_results.to_csv(f"{folder}/WAPC_full_develop_domain_metrics.csv")
# df.to_csv(f"{folder}/WAPC_full_develop_df.csv")

# %% [markdown]
# ---

# %%
GAZ_model = "segmenter_trained_with_ner_P_0.6-4901"
NewModel_metrics = pd.read_csv(f"{folder}/{GAZ_model}_domain_metrics.csv")
NewModel_metrics = NewModel_metrics.rename({"Unnamed: 0": "domain"}, axis=1).set_index('domain')
NewModel_metrics = NewModel_metrics.rename({"gt_tag_with_img": "GroundTruth-NewModel", "TP_pred": "TP_pred-NewModel", "FP_pred": "FP_pred-NewModel", "FN_pred": "FN_pred-NewModel"}, axis=1)

from ast import literal_eval
NewModel_metrics["GroundTruth-NewModel"] = NewModel_metrics["GroundTruth-NewModel"].apply(literal_eval)
NewModel_metrics["GroundTruth-NewModel"] = NewModel_metrics["GroundTruth-NewModel"].apply(set)

NewModel_metrics["TP_pred-NewModel"] = NewModel_metrics["TP_pred-NewModel"].apply(literal_eval)
NewModel_metrics["FP_pred-NewModel"] = NewModel_metrics["FP_pred-NewModel"].apply(literal_eval)

NewModel_metrics["Prediction-NewModel"] = NewModel_metrics.apply(lambda row: row["FP_pred-NewModel"].union(row["TP_pred-NewModel"] ), axis=1)
NewModel_metrics["Prediction-NewModel"] = NewModel_metrics["Prediction-NewModel"].apply(lambda row: sorted(list(set(row))))

pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)
NewModel_metrics

# %%
both_metrics = NewModel_metrics.join(Production_metrics, how = "outer")

# %%
both_metrics[["GroundTruth-Production", "GroundTruth-NewModel"]]

# %%
if both_metrics["GroundTruth-NewModel"].equals(both_metrics["GroundTruth-Production"]):
    both_metrics["GroundTruth"] = both_metrics["GroundTruth-Production"]


# %%
# pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

both_metrics

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
# both_metrics[["Prediction-NewModel", "Predictions-Production", "GroundTruth-Production"]]

# %%
both_metrics = both_metrics[["Prediction-NewModel", "Prediction-Production", "GroundTruth"]]


# %%
def create_confusion_matrix(df, pred_col):
    df[f"{pred_col.split('-')[-1]}_TP"] = df.apply(lambda row: [x for x in row[pred_col] if x in row["GroundTruth"]], axis=1)
    df[f"{pred_col.split('-')[-1]}_FP"] = df.apply(lambda row: [x for x in row[pred_col] if x not in row["GroundTruth"]], axis=1)
    df[f"{pred_col.split('-')[-1]}_FN"] = df.apply(lambda row: [x for x in row["GroundTruth"] if x not in row[pred_col]], axis=1)
    return df

both_metrics = create_confusion_matrix(both_metrics, "Prediction-NewModel")
both_metrics = create_confusion_matrix(both_metrics, "Prediction-Production")

# %%
both_metrics.to_csv("model_results/both_metrics.csv")
