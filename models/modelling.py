"""
UFI Project — Predictive Modelling
Trains and evaluates two models:
  1. Regression  — predict UFI score (continuous)
  2. Classification — predict UFI class (ordinal multi-class)

Models used:
  • Random Forest (primary, interpretable)
  • Gradient Boosting (XGBoost-style via sklearn)
  • Baseline Linear / Logistic Regression

Outputs:
  • Console metrics
  • Feature importance chart
  • Confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LinearRegression, LogisticRegression

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["C1", "C2", "C3", "C4", "hour", "speed_limit", "incident_count"]
TARGET_REG   = "ufi_score"
TARGET_CLASS = "ufi_class"

CLASS_ORDER = ["Free Flow", "Low", "Moderate", "High", "Severe"]


def prepare_data(df: pd.DataFrame):
    X = df[FEATURES].copy()

    # Regression target
    y_reg = df[TARGET_REG].copy()

    # Classification target (ordinal encode)
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_cls = le.transform(df[TARGET_CLASS])

    return X, y_reg, y_cls, le


# ── Regression ────────────────────────────────────────────────────────────────

def train_regression(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    print("\n── Regression Results ──")
    print(f"{'Model':<25}  {'MAE':>7}  {'R²':>7}")
    print("-" * 44)

    best_model, best_r2 = None, -np.inf
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        print(f"{name:<25}  {mae:>7.3f}  {r2:>7.4f}")
        if r2 > best_r2:
            best_r2, best_model = r2, (name, model)

    print(f"\nBest regressor: {best_model[0]}  (R² = {best_r2:.4f})")
    return best_model[1]


# ── Classification ────────────────────────────────────────────────────────────

def train_classification(X_train, X_test, y_train, y_test, le):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print("\n── Classification Results ──")
    best_model, best_acc = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = (preds == y_test).mean()
        print(f"{name:<25}  Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model = acc, (name, model)

    print(f"\nBest classifier: {best_model[0]}  (Acc = {best_acc:.4f})")
    best_preds = best_model[1].predict(X_test)
    label_names = [CLASS_ORDER[i] for i in sorted(set(y_test) | set(best_preds)) if i < len(CLASS_ORDER)]
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds,
                                 target_names=le.classes_,
                                 zero_division=0))
    return best_model[1], best_preds


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, title: str, filename: str):
    if not hasattr(model, "feature_importances_"):
        print(f"  (No feature importance for {title})")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)
    colors = ["#e74c3c" if importances[i] == max(importances) else "#3498db" for i in idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, le, filename: str):
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — UFI Class Prediction")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Actual vs Predicted ───────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, filename: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=12, color="#2980b9")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual UFI Score")
    ax.set_ylabel("Predicted UFI Score")
    ax.set_title("Actual vs Predicted UFI Score")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(model, X, y, cv=5, scoring="r2", label=""):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"  {label}  {scoring} CV ({cv}-fold): {scores.mean():.4f} ± {scores.std():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_modelling(df: pd.DataFrame):
    print("\n══════════════════════════════════════")
    print("   UFI PREDICTIVE MODELLING")
    print("══════════════════════════════════════")

    X, y_reg, y_cls, le = prepare_data(df)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # Regression
    best_reg = train_regression(X_tr, X_te, yr_tr, yr_te)
    plot_feature_importance(
        best_reg, FEATURES,
        "Feature Importance — UFI Score (Regression)",
        "08_reg_feature_importance.png"
    )
    plot_actual_vs_predicted(yr_te, best_reg.predict(X_te), "09_actual_vs_predicted.png")

    print("\nCross-validation (regression):")
    cross_validate(best_reg, X, y_reg, scoring="r2",             label="R²  ")
    cross_validate(best_reg, X, y_reg, scoring="neg_mean_absolute_error", label="MAE ")

    # Classification
    best_cls, cls_preds = train_classification(X_tr, X_te, yc_tr, yc_te, le)
    plot_feature_importance(
        best_cls, FEATURES,
        "Feature Importance — UFI Class (Classification)",
        "10_cls_feature_importance.png"
    )
    plot_confusion_matrix(yc_te, cls_preds, le, "11_confusion_matrix.png")

    print("\nCross-validation (classification):")
    cross_validate(best_cls, X, y_cls, scoring="accuracy", label="Acc ")
    cross_validate(best_cls, X, y_cls, scoring="f1_weighted", label="F1  ")

    return best_reg, best_cls


if __name__ == "__main__":
    df = pd.read_csv("data/ufi_scored.csv")
    run_modelling(df)
