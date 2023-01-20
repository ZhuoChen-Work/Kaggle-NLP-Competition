from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np 
import pandas as pd

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def calculate_overall_auc(df, label_col, predict_col):
    true_labels = df[label_col]>0.5
    predicted_labels = df[predict_col]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label_col, predict_col):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label_col]>0.5), subgroup_examples[predict_col])

def compute_bpsn_auc(df, subgroup, label_col, predict_col):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label_col]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label_col]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label_col]>0.5, examples[predict_col])

def compute_bnsp_auc(df, subgroup, label_col, predict_col):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label_col]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label_col]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label_col]>0.5, examples[predict_col])

def compute_bias_metrics_for_model(dataset, subgroups,label_col,predict_col,include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {'subgroup': subgroup,'subgroup_size': len(dataset[dataset[subgroup]>0.5])}
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, predict_col)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, predict_col)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, predict_col)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
