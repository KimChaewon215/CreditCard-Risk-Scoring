# common/evaluation.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def evaluate_classifier(y_true, y_pred, y_proba=None) -> dict:
    """
    기본 분류 지표 + (가능하면) ROC-AUC/PR-AUC까지 리턴
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0)
    }
    if y_proba is not None:
        # 이진분류 확률의 positive 클래스 열을 기대
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            pos = y_proba[:, 1]
        else:
            pos = y_proba
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, pos))
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, pos))
        except Exception:
            out["pr_auc"] = None
    return out