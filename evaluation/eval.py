import numpy as np
from sklearn import metrics
import torch

# https://www.kaggle.com/anokas/weighted-auc-metric-updated

def auc_eval_func(t, pred):
    y_true = np.clip(t, a_min=0, a_max=1).astype(int)
    #y_pred = 1 - np.exp(pred) / np.sum(np.exp(pred), axis=1)[:, np.newaxis]
    y_pred = 1 - torch.nn.functional.softmax(torch.from_numpy(np.array(pred)), dim=1).data.cpu().numpy()[:, 0]
    #y_pred = y_pred[:, 0]

    return alaska_weighted_auc(y_true, y_pred)


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2,   1]

    #print(y_valid)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization
