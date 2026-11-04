"""
Iris Dataset Classification & Comprehensive Evaluation Metrics
AI and Expert Systems Assignment 1

This script loads the Iris dataset from scikit-learn, trains a classifier,
and computes all major classification evaluation metrics for multi-class classification:
  - Accuracy & Error Rate
  - Confusion Matrix
  - Precision (Macro, Micro, Weighted, Per-Class)
  - Recall / Sensitivity (Macro, Micro, Weighted, Per-Class)
  - Specificity (Per-Class & Macro Average)
  - F1-Score (Macro, Micro, Weighted, Per-Class)
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa Score
  - Log Loss (Cross-Entropy Loss)
  - Multi-class ROC AUC (One-vs-Rest and One-vs-One)
  - Full Classification Report
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    roc_curve,
    auc,
)


def calculate_per_class_specificity(cm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate One-vs-Rest Specificity (True Negative Rate) for each class
    from a multi-class confusion matrix.
    Specificity = TN / (TN + FP)
    """
    n_classes = cm.shape[0]
    total_samples = np.sum(cm)
    specificities = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fp + fn)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    specificities = np.array(specificities)
    macro_specificity = float(np.mean(specificities))
    return specificities, macro_specificity


def run_iris_classification():
    print("=" * 70)
    print("        IRIS DATASET MULTI-CLASS CLASSIFICATION BENCHMARK        ")
    print("=" * 70)

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    n_classes = len(target_names)

    print(f"\n[+] Dataset loaded successfully:")
    print(f"    - Total instances: {X.shape[0]}")
    print(f"    - Features: {X.shape[1]} -> {', '.join(feature_names)}")
    print(f"    - Classes: {n_classes} -> {list(target_names)}")

    # 2. Train-Test Split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples:  {len(X_test)}")

    # 3. Model Training
    # We use a Random Forest Classifier (robust, highly interpretable)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # 4. Predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

    # 5. Compute Metrics
    # A. Accuracy & Error Rate
    acc = accuracy_score(y_test, y_pred)
    error_rate = 1.0 - acc

    # B. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # C. Precision
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_micro = precision_score(y_test, y_pred, average="micro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)

    # D. Recall / Sensitivity
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_micro = recall_score(y_test, y_pred, average="micro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)

    # E. Specificity (True Negative Rate)
    spec_per_class, spec_macro = calculate_per_class_specificity(cm)

    # F. F1-Score
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # G. Advanced Correlation & Agreement Metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    # H. Multi-class ROC AUC
    roc_auc_ovr_macro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    roc_auc_ovr_weighted = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    roc_auc_ovo_macro = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    roc_auc_ovo_weighted = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")

    # 6. Display Formatted Evaluation Results
    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS SUMMARY                   ")
    print("=" * 70)

    print(f"\n1. GLOBAL ACCURACY & LOSS:")
    print(f"   - Accuracy:                  {acc * 100:.2f}% ({acc:.4f})")
    print(f"   - Error Rate:                {error_rate * 100:.2f}% ({error_rate:.4f})")
    print(f"   - Log Loss (Cross-Entropy):  {loss:.4f}")

    print(f"\n2. PRECISION METRICS:")
    print(f"   - Macro Average:             {prec_macro:.4f}")
    print(f"   - Micro Average:             {prec_micro:.4f}")
    print(f"   - Weighted Average:          {prec_weighted:.4f}")

    print(f"\n3. RECALL / SENSITIVITY METRICS:")
    print(f"   - Macro Average:             {rec_macro:.4f}")
    print(f"   - Micro Average:             {rec_micro:.4f}")
    print(f"   - Weighted Average:          {rec_weighted:.4f}")

    print(f"\n4. SPECIFICITY (TRUE NEGATIVE RATE):")
    print(f"   - Macro Average:             {spec_macro:.4f}")

    print(f"\n5. F1-SCORE (HARMONIC MEAN):")
    print(f"   - Macro Average:             {f1_macro:.4f}")
    print(f"   - Micro Average:             {f1_micro:.4f}")
    print(f"   - Weighted Average:          {f1_weighted:.4f}")

    print(f"\n6. CORRELATION & AGREEMENT:")
    print(f"   - Matthews Corr. Coeff (MCC):{mcc:.4f}")
    print(f"   - Cohen's Kappa:             {kappa:.4f}")

    print(f"\n7. AREA UNDER ROC CURVE (ROC AUC):")
    print(f"   - One-vs-Rest (Macro):       {roc_auc_ovr_macro:.4f}")
    print(f"   - One-vs-Rest (Weighted):    {roc_auc_ovr_weighted:.4f}")
    print(f"   - One-vs-One  (Macro):       {roc_auc_ovo_macro:.4f}")
    print(f"   - One-vs-One  (Weighted):    {roc_auc_ovo_weighted:.4f}")

    print("\n8. PER-CLASS DETAILED BREAKDOWN:")
    per_class_df = pd.DataFrame({
        "Class": target_names,
        "Precision": prec_per_class,
        "Recall (Sensitivity)": rec_per_class,
        "Specificity": spec_per_class,
        "F1-Score": f1_per_class,
        "Support": np.bincount(y_test, minlength=n_classes)
    })
    print(per_class_df.to_string(index=False))

    print("\n9. CONFUSION MATRIX:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {name}" for name in target_names],
                             columns=[f"Pred {name}" for name in target_names])
    print(cm_df.to_string())

    print("\n10. STANDARD CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 7. Generate Visualizations (Confusion Matrix & ROC Curves)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    # A. Confusion Matrix Plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - Iris Classification")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_plot_path}")

    # B. Multi-Class ROC Curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"ROC curve: {target_names[i]} (AUC = {class_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Multi-Class ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"[+] ROC curves plot saved to: {roc_plot_path}")

    print("\n" + "=" * 70)
    print("                 BENCHMARK COMPLETED SUCCESSFULLY                ")
    print("=" * 70)


if __name__ == "__main__":
    run_iris_classification()
