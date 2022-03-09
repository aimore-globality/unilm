# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: wae
#     language: python
#     name: python3
# ---

# # Compare WAPC in production with a MarkUpLM

# + tags=[]
import pandas as pd
from web_annotation_extractor.bundles.past_client.bundle import PastClientBundle
from microcosm.api import create_object_graph
from microcosm_sagemaker.bundle import Bundle, BundleInputArtifact, BundleOutputArtifact
# -

# ## Load the SWDE-CF data

# + tags=[]
import pandas as pd
from pathlib import Path
dataset = 'develop'

# + tags=[]
dataset

# + tags=[]
if dataset == 'develop':
    data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1735)_neg(4035)_intermediate.pkl"

# + tags=[]
df = pd.read_pickle(data_path)

# + tags=[]
pd.set_option('max_columns',50, 'max_colwidth',20)
print(len(df))
df.head(2)
# -

# ## Load the WAPC model

# + tags=[]
graph = create_object_graph('test')
pc = PastClientBundle(graph)

# + tags=[]
pc.load(BundleInputArtifact("../../../globality-ml-scripts/wae_models/model/past_client_bundle/"))
# -

# ## Predict WAPC on this data

# + tags=[]
df['extracted_entities'] = pc.predict_batch(df['url'], df['html'], df['content_type'])

# + tags=[]
df
# -

# ## Get WAPC metrics

# + tags=[]
from web_annotation_extractor.evaluations.exact_match_evaluation import ExactMatchEvaluation

# + tags=[]
ground_truth = df['annotations-PAST_CLIENT'].dropna().apply(lambda x: [str(y.get("value")) for y in x if y.get("value") != None])
df["ground_truth"] = ground_truth

# + tags=[]
predictions = df["extracted_entities"].apply(lambda x: [str(w) for z in [y.taxonomy_uris for y in x] for w in z])
df["predictions"] = predictions


# + tags=[]
# Example:
# p = {'a', 'b', 'c', '1', '2'}
# gt = {'b', 'c', 'd'}

# TP = p & gt
# FP = p - gt
# FN = gt - p

# print(f"TP: {TP} - {len(TP)}, \
#         FP: {FP} - {len(FP)}, \
#         FN: {FN} - {len(FN)}")

# + tags=[]
def compute_recall(ground_truth, prediction):
    prediction_set = {str(text) for text in predictions}
    ground_truth_set = {str(text) for text in ground_truth}
    
    TP = len(prediction_set & ground_truth_set)
    FN = len(ground_truth_set - prediction_set)
    
    if TP+FN > 0:
        return TP/(TP+FN)
    return None

def compute_precision_adjusted(ground_truth, prediction, negative_percentage=1):
    positive_index = ground_truth.dropna()[ground_truth.dropna().apply(len) > 0].index
    ground_truth_pos = ground_truth.loc[positive_index]
    prediction_pos = prediction.loc[positive_index]

    ground_truth_neg = ground_truth.loc[~ground_truth.isin(ground_truth_pos)]
    prediction_neg = prediction.loc[~prediction.isin(ground_truth_pos)]

    fp_scalar = 1/negative_percentage

    prediction_set = {str(text) for text in prediction}
    ground_truth_set = {str(text) for text in ground_truth}
    TP = len(prediction_set & ground_truth_set)
    FP = len(prediction_set - ground_truth_set)
    
    prediction_set_pos = {str(text) for text in prediction_pos}
    ground_truth_set_pos = {str(text) for text in ground_truth_pos}

    prediction_set_neg = {str(text) for text in prediction_neg}
    ground_truth_set_neg = {str(text) for text in ground_truth_neg}

    FP_pos = len(prediction_set_pos - ground_truth_set_pos)
    FP_neg = len(prediction_set_neg - ground_truth_set_neg) * fp_scalar
    FP_adjusted = FP_pos + FP_neg
    
    # print(f" TP: {TP} \n FP: {FP} \n FP_neg: {FP_neg} \n FP_pos: {FP_pos} \n FP_adjusted: {FP_adjusted}")

    if TP + FP_adjusted > 0:        
        precision_adjusted = TP/(TP + FP_adjusted)
        return precision_adjusted
    return None
    

negative_percentage = 0.10

def compute_segmentation_metrics(ground_truth_series, predictions_series):
    precision = compute_precision_adjusted(ground_truth_series, predictions_series, negative_percentage)
    recall = compute_recall(ground_truth_series, predictions_series)

    # recall = ExactMatchEvaluation.compute_recall(ground_truth, predictions)
    # precision = ExactMatchEvaluation.compute_precision(ground_truth, predictions)
    # print(precision, recall)
    f1 = ExactMatchEvaluation.compute_f1(precision, recall)
    return pd.Series({
            "segmentation_precision": precision,
            "segmentation_recall": recall,
            "segmentation_f1": f1,
        })


# + tags=[]
df_domain_metrics = df.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.predictions))
df_domain_metrics.mean().to_dict()
# -

df_domain_metrics = df.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.predictions))
df_domain_metrics.mean().to_dict()

# # Load SWDE data

# # Load MarkupLM Trained Model and predict on something 

# +
# from markuplmft.fine_tuning.run_swde.lm_model import LModel
# lm = LModel()
# lm.load_data()
# lm.prepare_model_to_train()
# lm.fit()
# -

from markuplmft.fine_tuning.run_swde.lm_model import LModel
lm = LModel()
lm.load_data('develop')
results = lm.predict_on_develop()

lm.evaluate(results)

# ## Load the trained MarkUp model 

# + tags=[]
pc = PastClientBundle(graph)

# + tags=[]
df = pc.simplifier.simplify(df)

# + tags=[]
df = pc.segmenter_url.segment(df)
df = pc.segmenter_html.segment(df)

# + tags=[]
html_predictions_without_training = df['html_simplified_segmentations'].apply(lambda x: [pc.uri_company_map.get(y[0]) for y in x])

# + tags=[]
url_predictions_without_training = df['url_simplified_segmentations'].apply(lambda x: [pc.uri_company_map.get(y[0]) for y in x])

# + tags=[]
predictions_without_training = html_predictions_without_training + url_predictions_without_training

# + tags=[]
df["predictions_without_training"] = predictions_without_training
df_domain_metrics = df.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.predictions_without_training))
df_domain_metrics.mean().to_dict()
# -




# ## Predict the MarkUp model on the data

pc = PastClientBundle(graph)
pc

# ## Get MarkupLM metrics



# ## Compare Results


