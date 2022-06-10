from sklearn.metrics import confusion_matrix
import numpy as np
from markuplmft.fine_tuning.run_swde import constants
from typing import Sequence

def compute_metrics(truth:Sequence[str], pred:Sequence[str]):
    """
    truth: ['none', 'none', 'tag']
    pred: ['none', 'tag', 'tag']
    """
    metrics = {}
    
    truth = np.array(truth)
    pred = np.array(pred)

    cm = confusion_matrix(truth, pred, labels=constants.ATTRIBUTES_PLUS_NONE)
    cm = {
        'TP': cm[0, 0], 
        'FN': cm[0, 1],
        'FP': cm[1, 0], 
        'TN': cm[1, 1]
        }

    precision = cm["TP"] / (cm["TP"] + cm["FP"])
    recall = cm["TP"] / (cm["TP"] + cm["FN"])
    f1 = 2 * (precision * recall) / (precision + recall)

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return metrics, cm

def compute_metrics_per_dataset(df):
    groung_truth = df['node_gt_tag']
    predictions = df['node_pred_tag']
    return compute_metrics(groung_truth, predictions)


# def compute_metrics_per_domain(df):
#     df.groupby('domain')
#     return compute_metrics(groung_truth, predictions)

# def compute_metrics_per_page(df):
#     df.groupby('html_path')
#     return compute_metrics(groung_truth, predictions)