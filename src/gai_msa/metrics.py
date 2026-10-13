import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from .constants import LABELS


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


def full_reports(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(LABELS)))
    )
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    return report_df, cm_df
