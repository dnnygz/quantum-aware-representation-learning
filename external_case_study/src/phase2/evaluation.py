import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, accuracy_score, log_loss, confusion_matrix)


def metrics(y, probability):
    p = np.asarray(probability)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions')
    return dict(roc_auc=float(roc_auc_score(y, p)),
                ap_er_positive=float(average_precision_score(y, p)),
                ap_er_negative=float(average_precision_score(1-y, 1-p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                accuracy=float(accuracy_score(y, p >= .5)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                confusion_matrix=confusion_matrix(y, p >= .5, labels=[0, 1]).tolist())
