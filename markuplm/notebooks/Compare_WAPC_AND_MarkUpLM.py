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
from microcosm_sagemaker.bundle import BundleInputArtifact
from web_annotation_extractor.evaluations.exact_match_evaluation import ExactMatchEvaluation
from microcosm.api import create_object_graph
graph = create_object_graph('test')

pd.set_option('max_columns',60, 'max_colwidth',80, 'max_rows',5)
# -

# ## Load the SWDE-CF data

# + tags=[]
dataset = 'develop'
print(dataset)
if dataset == 'develop':
    data_path = f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1735)_neg(4035)_intermediate.pkl"
df = pd.read_pickle(data_path)

print(len(df))
df.head(2)
# -

df['domain'].value_counts()

values = df['annotations'].apply(lambda x: x.get("PAST_CLIENT")).dropna().apply(lambda annotations: [x['value'] for x in annotations if x['value']]) 
values = values[values.apply(len) > 0]

taxonomy_to_text_mappings = dict([(company.uri, company.name)for company in graph.known_company_taxonomy])
df['values'] = values
df['values'] = df['values'].fillna('').apply(list)
df["value-PAST_CLIENT"] = df['values'].apply(lambda values: [taxonomy_to_text_mappings.get(x) for x in values]) 
df

# ## Load and Predict WAPC model

# + tags=[]
graph = create_object_graph('test')
pc = PastClientBundle(graph)
pc.load(BundleInputArtifact("../../../globality-ml-scripts/wae_models/model/past_client_bundle/"))

# + tags=[]
df['extracted_entities'] = pc.predict_batch(df['url'], df['html'], df['content_type'])

# + tags=[]
df['extracted_entities'].dropna().apply(len).value_counts()
# -

df["value-PAST_CLIENT"].dropna().apply(len).value_counts()

# ## Get WAPC metrics

# + tags=[]
df["ground_truth"] = df["values"]
df['ground_truth'] = df['ground_truth'].fillna('').apply(list)
predictions = df["extracted_entities"].apply(lambda x: [str(w) for z in [y.taxonomy_uris for y in x] for w in z])
df["predictions"] = predictions

# + tags=[]
debug = False
negative_percentage = 0.10

def compute_recall(ground_truth, prediction):
    prediction_set = {past_client for past_clients in prediction.values for past_client in past_clients}
    ground_truth_set = {past_client for past_clients in ground_truth.values for past_client in past_clients}
    
    TP = len(prediction_set & ground_truth_set)
    FN = len(ground_truth_set - prediction_set)
    if debug: 
        cf_values = pd.DataFrame(
            dict(
                TP=TP,
                FN=FN,
            ).items()
            ).set_index(0).T

        display(cf_values)

    if TP+FN > 0:
        return TP/(TP+FN)
    return None

def compute_precision_adjusted(ground_truth, prediction, negative_percentage=1):
    assert len(ground_truth) == len(prediction)
    positive_index = ground_truth[ground_truth.apply(len) > 0].index
    pos_ground_truth = ground_truth.loc[positive_index]
    pos_prediction = prediction.loc[positive_index]

    neg_ground_truth = ground_truth.loc[~ground_truth.isin(pos_ground_truth)]
    neg_prediction = prediction.loc[~prediction.isin(pos_ground_truth)]

    fp_scalar = 1/negative_percentage

    prediction_set = {past_client for past_clients in prediction.values for past_client in past_clients}
    ground_truth_set = {past_client for past_clients in ground_truth.values for past_client in past_clients}

    pos_prediction_set = {past_client for past_clients in pos_prediction.values for past_client in past_clients}
    pos_ground_truth_set = {past_client for past_clients in pos_ground_truth.values for past_client in past_clients}
    
    neg_prediction_set = {past_client for past_clients in neg_prediction.values for past_client in past_clients}
    neg_ground_truth_set = {past_client for past_clients in neg_ground_truth.values for past_client in past_clients}

    TP = len(prediction_set & ground_truth_set)
    
    FP = len(prediction_set - ground_truth_set)

    FP_pos = len(pos_prediction_set - pos_ground_truth_set)
    FP_neg = len(neg_prediction_set - neg_ground_truth_set) 
    FP_adjusted = FP_pos + FP_neg * fp_scalar
    if debug:
        cf_values = pd.DataFrame(
            dict(
                TP=TP,
                FP=FP,
                FP_neg=FP_neg,
                FP_pos=FP_pos,
                FP_adjusted=FP_adjusted
            ).items()
            ).set_index(0).T

        display(cf_values)

    if TP + FP_adjusted > 0:        
        precision_adjusted = TP/(TP + FP_adjusted)
        return precision_adjusted
    return None

def compute_segmentation_metrics(ground_truth_series, prediction_series, domain=''):
    """
    Input
        ground_truth_series: pd.Series with a list of groundtruth strings
        predictions_series: pd.Series with a list of predited strings
    Output
        pd.Series with Precision, Recalla and F1 metrics 
    """
    if debug:
        print(domain.iloc[0])
    precision = compute_precision_adjusted(ground_truth_series, prediction_series, negative_percentage)
    recall = compute_recall(ground_truth_series, prediction_series)

    # recall = ExactMatchEvaluation.compute_recall(ground_truth, predictions)
    # precision = ExactMatchEvaluation.compute_precision(ground_truth, predictions)
    # print(precision, recall)
    f1 = ExactMatchEvaluation.compute_f1(precision, recall)
    results = pd.Series(dict(
            segmentation_precision = precision,
            segmentation_recall = recall,
            segmentation_f1 = f1
       ))
    if debug:
        display(results)
    return results


# + tags=[]
df_domain_metrics = df.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.predictions, x.domain))
df_domain_metrics.mean().to_dict()
# -

df_domain_metrics

# +
debug = False
negative_percentage = 0.10

def compute_f1(precision, recall):
    if precision and recall:
        return 2*(precision * recall) / (precision + recall)
    else:
        return None

def compute_precision(TP, FP):
    if TP + FP > 0:        
        return TP/(TP + FP)
    else:
        return None

def compute_recall(TP, FN):
    if TP + FN > 0:        
        return TP/(TP + FN)
    else:
        return None

def compute_cf_matrix(ground_truth, prediction):
    assert len(ground_truth) == len(prediction)
    positive_index = ground_truth[ground_truth.apply(len) > 0].index
    pos_ground_truth = ground_truth.loc[positive_index]
    pos_prediction = prediction.loc[positive_index]

    neg_ground_truth = ground_truth.loc[~ground_truth.isin(pos_ground_truth)]
    neg_prediction = prediction.loc[~prediction.isin(pos_ground_truth)]

    fp_scalar = 1/negative_percentage

    prediction_set = {past_client for past_clients in prediction.values for past_client in past_clients}
    ground_truth_set = {past_client for past_clients in ground_truth.values for past_client in past_clients}

    pos_prediction_set = {past_client for past_clients in pos_prediction.values for past_client in past_clients}
    pos_ground_truth_set = {past_client for past_clients in pos_ground_truth.values for past_client in past_clients}
    
    neg_prediction_set = {past_client for past_clients in neg_prediction.values for past_client in past_clients}
    neg_ground_truth_set = {past_client for past_clients in neg_ground_truth.values for past_client in past_clients}

    TP = len(prediction_set & ground_truth_set)
    
    FP = len(prediction_set - ground_truth_set)

    FN = len(ground_truth_set - prediction_set)

    FP_pos = len(pos_prediction_set - pos_ground_truth_set)
    FP_neg = len(neg_prediction_set - neg_ground_truth_set) 
    FP_adjusted = FP_pos + FP_neg * fp_scalar

    precision = compute_precision(TP, FP)
    precision_adjusted = compute_precision(TP, FP_adjusted)
    recall = compute_recall(TP, FN)
    f1 = compute_f1(TP, FN)
    num_positives = TP + FN
    
    results = dict(
        TP=TP, 
        FP=FP, 
        FN=FN,
        num_positives=num_positives,
        FP_adjusted=FP_adjusted,
        precision=precision,
        precision_adjusted=precision_adjusted,
        recall=recall,
        f1=f1,
        )

    return results

cf_matrix = dict()
for domain_name, domain_df in df.groupby("domain"):
    cf_matrix[domain_name] = compute_cf_matrix(domain_df["ground_truth"], domain_df["predictions"])

cf_matrix = pd.DataFrame(cf_matrix).T.sort_values("num_positives")
len(cf_matrix)
# -

cf_matrix = cf_matrix[cf_matrix["num_positives"] > 0]
cf_matrix

# +
import plotly.graph_objects as go


def plot_performance(
    domain,
    TP,
    FP,
    FN,
    num_positives,
    precision,
    recall,
    f1,
    title="Performance per Domain",
):
    trace_TP = go.Bar(
        x=domain,
        y=TP,
        name="TP",
        yaxis="y1",
    )
    trace_FP = go.Bar(
        x=domain,
        y=FP,
        name="FP",
        yaxis="y1",
    )

    trace_FN = go.Bar(
        x=domain,
        y=FN,
        name="FN",
        yaxis="y1",
    )
    trace_pos = go.Bar(
        x=domain,
        y=num_positives,
        name="num_positives",
        yaxis="y1",
    )

    trace_precision = go.Scatter(
        x=domain,
        y=100 * precision,
        name="precision",
        mode="lines+markers",
        marker=dict(
            size=6,
            color="Red",
            symbol="cross",
            line_color='red',
            line_width=1,
        ),
        yaxis="y2",
    )

    trace_recall = go.Scatter(
        x=domain,
        y=100 * recall,
        name="recall",
        mode="lines+markers",
        yaxis="y2",
    )

    data = [trace_TP, trace_FP, trace_FN, trace_pos, trace_precision, trace_recall]

    layout = go.Layout(
        title=title,
        yaxis=dict(
            title="Counts",
        ),
        yaxis2=dict(
            title="%",
            overlaying="y",
            side="right",
            range=[-5, 105],
        ),
        width=len(domain) * 27 if len(domain) > 20 else 600,
        height=600,
        margin=dict(l=5, r=5, b=20, t=50, pad=1),
        legend=dict(
            x=0.007,
            y=0.97,
            bordercolor="Black",
            borderwidth=1,
        ),
    )
    return go.Figure(data=data, layout=layout)



# -

plot_performance(
    domain=cf_matrix.index,
    TP=cf_matrix["TP"],
    FP=cf_matrix["FP"],
    FN=cf_matrix["FN"],
    num_positives=cf_matrix["num_positives"],
    precision=cf_matrix["precision"],
    recall=cf_matrix["recall"],
    f1=cf_matrix["f1"],
        )

plot_performance(
    title="Performance per Domain Adjusted",
    domain=cf_matrix.index,
    TP=cf_matrix["TP"],
    FP=cf_matrix["FP_adjusted"],
    FN=cf_matrix["FN"],
    num_positives=cf_matrix["num_positives"],
    precision=cf_matrix["precision_adjusted"],
    recall=cf_matrix["recall"],
    f1=cf_matrix["f1"],
        )

cf_matrix


# +
def compute_metrics_for_dataset(cf_matrix):
    cf_matrix_all = cf_matrix[["TP", "FP", "FN", "num_positives", "FP_adjusted"]].sum()
    cf_matrix_all['precision'] = compute_precision(TP=cf_matrix_all['TP'], FP=cf_matrix_all['FP'])
    cf_matrix_all['precision_adjusted'] = compute_precision(TP=cf_matrix_all['TP'], FP=cf_matrix_all['FP_adjusted'])
    cf_matrix_all['recall'] = compute_recall(TP=cf_matrix_all['TP'], FN=cf_matrix_all['FN'])
    cf_matrix_all['f1'] = compute_f1(precision=cf_matrix_all['precision'], recall=cf_matrix_all['recall'])
    cf_matrix_all['domain'] = "All"
    cf_matrix_all = pd.DataFrame(cf_matrix_all).T
    cf_matrix_all = cf_matrix_all.set_index('domain').reset_index()
    cf_matrix_all
    return cf_matrix_all


# -

cf_matrix_all = compute_metrics_for_dataset(cf_matrix)
cf_matrix_all

plot_performance(
    title="Performance per Domain Adjusted",
    domain=cf_matrix_all.index,
    TP=cf_matrix_all["TP"],
    FP=cf_matrix_all["FP_adjusted"],
    FN=cf_matrix_all["FN"],
    num_positives=cf_matrix_all["num_positives"],
    precision=cf_matrix_all["precision_adjusted"],
    recall=cf_matrix_all["recall"],
    f1=cf_matrix_all["f1"],
        )

plot_performance(
    title="Performance per Domain",
    domain=cf_matrix_all.index,
    TP=cf_matrix_all["TP"],
    FP=cf_matrix_all["FP"],
    FN=cf_matrix_all["FN"],
    num_positives=cf_matrix_all["num_positives"],
    precision=cf_matrix_all["precision"],
    recall=cf_matrix_all["recall"],
    f1=cf_matrix_all["f1"],
        )

# # Load MarkupLM Trained Model and predict on something 

# ---

# +
import pandas as pd

results = pd.read_pickle("results_classified_5_epoch.pkl")
# -

results

results = results.reset_index().drop('index', axis=1)
pos_results = results[results['pred_type']=='PAST_CLIENT'] #! The Performance of the MarkupLM
# pos_results = results[results['truth']=='PAST_CLIENT'] #! The best this system can give, considering that the annotations for the text are correct. However, this might not be the best for the values.
# pos_results = results #! No classification 
len(pos_results)

sementation_on_nodes  = pc.segmenter_html._segment_companies(pos_results['text']).fillna("").apply(list)
results['sementation_on_nodes'] = sementation_on_nodes
results['reconciliations'] = pc.get_reconciliations(results['sementation_on_nodes'].dropna())
results['reconciliations'].dropna().sort_values()

# +
# #? Create mapping to convert value into taxonomy
value_to_taxonomy_mappings = dict([(company.name, company.uri)for company in graph.known_company_taxonomy])

# #? Convert reconciliations into reconciliations taxonomized 
results['reconciliations_taxonomized'] = results['reconciliations'].dropna().apply(lambda values: [value_to_taxonomy_mappings.get(x) for x in values]) 

# #? Create the domain column
results['domain'] = results['domain'].apply(lambda x: x.split('.pickle')[0])

# #? Get the intersection of the domains
df = df[df['domain'].isin(results['domain'])]

# #? Create mapping to convert value into taxonomy 
results_grouped = pd.DataFrame(results.groupby(by='html_path').agg({'reconciliations_taxonomized': lambda x: [z for y in list(x.dropna()) for z in y] , 'domain': lambda x: list(x)[0]}))

# #? Make sure both datasets have the same number of pages
assert len(df) == len(results_grouped)
# -

# #? Load and apply pageid to url mapping
pageid_url_mapping = pd.read_pickle("/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl")
results_grouped.reset_index(inplace=True)
results_grouped['url'] = results_grouped['html_path'].apply(lambda x: pageid_url_mapping.get(x)[0])

results_grouped = results_grouped.set_index("url")
df = df.set_index("url")

merge = df.join(results_grouped, lsuffix="_l")

# #! Model in Production
merge_domain_metrics = merge.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.predictions, x.domain))
merge_domain_metrics.mean().to_dict()

# #! Positive Nodes are predictions by the model trained on 5 epochs
merge_domain_metrics = merge.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.reconciliations_taxonomized, x.domain))
merge_domain_metrics.mean().to_dict()

# #! Positive Nodes are groundtruth
merge_domain_metrics = merge.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.reconciliations_taxonomized, x.domain))
merge_domain_metrics.mean().to_dict()

# #! No classification 
merge_domain_metrics = merge.groupby("domain").apply(lambda x: compute_segmentation_metrics(x.ground_truth, x.reconciliations_taxonomized, x.domain))
merge_domain_metrics.mean().to_dict()

merge_domain_metrics.dropna().sort_values("segmentation_f1").plot()

merge_domain_metrics.to_csv("merge_domain_metrics_5_epochs.csv")

merge_domain_metrics = pd.read_csv("merge_domain_metrics_5_epochs.csv")

df_domain_metrics.to_csv("merge_domain_metrics_prod.csv")

df_domain_metrics[['segmentation_f1', "segmentation_precision", "segmentation_recall"]].plot.bar(figsize=(30, 5))

merge_domain_metrics[['segmentation_f1', "segmentation_precision", "segmentation_recall"]].plot.bar(figsize=(30, 5))

# ---

all_values = merge['annotations'].apply(lambda x: [y.get('value') for y in x.get("PAST_CLIENT", {} ) if y.get('value') ])
all_text = merge['annotations'].apply(lambda x: [y.get('text') for y in x.get("PAST_CLIENT", {} ) if y.get('text') ])
print(len([x for y in all_values for x in y]))
print(len([x for y in all_text for x in y]))
merge['all_values'] = all_values
merge['all_text'] = all_text
merge['contain_value'] = merge["all_values"].apply(len) > 0
merge['contain_text'] = merge["all_text"].apply(len) > 0

merge
