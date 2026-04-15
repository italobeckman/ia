import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            else:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            metrics["ROC-AUC"] = 0.0
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2_score(y_true, y_pred)
    }

def plot_confusion_matrix_and_save(y_true, y_pred, filepath="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_roc_and_pr_curves_and_save(y_true, y_prob, filepath_roc="roc_curve.png", filepath_pr="pr_curve.png"):
    if len(np.unique(y_true)) == 2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(filepath_roc)
        plt.close()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(filepath_pr)
        plt.close()
        return filepath_roc, filepath_pr
    return None, None

def plot_regression_residuals_and_save(y_true, y_pred, filepath="residuals.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Errors vs. Predicted')
    
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_training_history_and_save(train_losses, val_losses, train_scores, val_scores, filepath="training_history.png", score_label="Metric"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_scores, label=f"Train {score_label}")
    plt.plot(val_scores, label=f"Validation {score_label}")
    plt.xlabel('Epoch')
    plt.ylabel(score_label)
    plt.title(f'{score_label} Evolution')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_feature_importance_and_save(model, feature_names, filepath="feature_importance.png", X_reference=None, y_reference=None, random_state=42, n_repeats=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif X_reference is not None and y_reference is not None:
        try:
            result = permutation_importance(
                model,
                X_reference,
                y_reference,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            importances = result.importances_mean
        except Exception:
            return None
    else:
        return None

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath
