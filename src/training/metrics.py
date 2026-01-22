import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if y_prob is not None else None
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

def confidence_interval(metric, n, z=1.96):
    se = np.sqrt(metric * (1 - metric) / n)
    return metric - z * se, metric + z * se

def p_value(metric1, metric2, n1, n2):
    p = (metric1 * n1 + metric2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    z = (metric1 - metric2) / se
    return 2 * (1 - torch.distributions.Normal(0, 1).cdf(abs(z)))
