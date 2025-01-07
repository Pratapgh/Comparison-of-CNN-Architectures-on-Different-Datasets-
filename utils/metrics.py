
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, and F1-score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

