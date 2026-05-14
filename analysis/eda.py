"""
UFI Project — Data Analysis Module
Exploratory Data Analysis (EDA), statistical summaries, and
insight generation on the scored UFI dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = "RdYlGn_r"


# ── 1. Summary Statistics ──────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ufi_score", "C1", "C2", "C3", "C4", "avg_speed", "volume"]
    stats = df[cols].describe().T
    stats["median"]  = df[cols].median()
    stats["skew"]    = df[cols].skew()
    stats["kurtosis"]= df[cols].kurtosis()
    print("\n── Summary Statistics ──")
    print(stats[["mean", "median", "std", "min", "max", "skew"]].round(3).to_string())
    return stats


# ── 2. Neighbourhood-level Aggregation ────────────────────────────────────────

def neighbourhood_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("neighbourhood").agg(
        avg_ufi       =("ufi_score", "mean"),
        peak_ufi      =("ufi_score", "max"),
        avg_C1        =("C1", "mean"),
        avg_C2        =("C2", "mean"),
        avg_C3        =("C3", "mean"),
        avg_C4        =("C4", "mean"),
        incident_rate =("incident_count", "mean"),
    ).sort_values("avg_ufi", ascending=False).round(3)

    print("\n── Neighbourhood Summary (sorted by avg UFI) ──")
    print(agg.to_string())
    return agg


# ── 3. Hourly Trend ───────────────────────────────────────────────────────────

def plot_hourly_ufi(df: pd.DataFrame):
    hourly = df.groupby("hour")["ufi_score"].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hourly.index, hourly.values, alpha=0.3, color="#e74c3c")
    ax.plot(hourly.index, hourly.values, color="#c0392b", linewidth=2.5, marker="o", markersize=5)

    # Rush-hour shading
    for (lo, hi), label, color in [
        ((7, 10),  "AM Peak",   "#fdcb6e"),
        ((17, 20), "PM Peak",   "#fdcb6e"),
        ((12, 14), "Lunch",     "#dfe6e9"),
    ]:
        ax.axvspan(lo, hi, alpha=0.25, color=color, label=label)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean UFI Score")
    ax.set_title("City-wide UFI by Hour — Morning & Evening Peaks Clearly Visible")
    ax.set_xticks(range(24))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUTPUT_DIR / "01_hourly_ufi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 4. Neighbourhood Bar Chart ────────────────────────────────────────────────

def plot_neighbourhood_bar(df: pd.DataFrame):
    peak = df[df["temporal_weight"] == 1.0].groupby("neighbourhood")["ufi_score"].mean().sort_values()

    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(peak)))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(peak.index, peak.values, color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.axvline(55, linestyle="--", color="#e74c3c", linewidth=1.5, label="HIGH threshold (55)")
    ax.set_xlabel("Mean UFI Score (peak hours)")
    ax.set_title("Neighbourhood UFI at Peak Hour")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "02_neighbourhood_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 5. Component Heatmap ──────────────────────────────────────────────────────

def plot_component_heatmap(df: pd.DataFrame):
    comp = df.groupby("neighbourhood")[["C1", "C2", "C3", "C4"]].mean()
    comp.columns = ["C1 Link Perf.", "C2 Vol/Cap", "C3 Net Stress", "C4 NLP Sev."]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(comp, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Score"})
    ax.set_title("UFI Component Breakdown by Neighbourhood")
    fig.tight_layout()
    path = OUTPUT_DIR / "03_component_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 6. Correlation Matrix ─────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame):
    cols = ["ufi_score", "C1", "C2", "C3", "C4", "avg_speed", "volume", "incident_count"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    path = OUTPUT_DIR / "04_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 7. UFI Distribution ───────────────────────────────────────────────────────

def plot_ufi_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogram
    axes[0].hist(df["ufi_score"], bins=40, color="#3498db", edgecolor="white")
    axes[0].set_xlabel("UFI Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("UFI Score Distribution")

    # Class pie
    class_counts = df["ufi_class"].value_counts()
    class_order  = ["Free Flow", "Low", "Moderate", "High", "Severe"]
    class_counts = class_counts.reindex([c for c in class_order if c in class_counts], fill_value=0)
    colors_pie   = ["#27ae60", "#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    axes[1].pie(class_counts, labels=class_counts.index, autopct="%1.1f%%",
                colors=colors_pie[:len(class_counts)], startangle=140)
    axes[1].set_title("UFI Class Distribution")

    fig.tight_layout()
    path = OUTPUT_DIR / "05_ufi_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 8. Speed vs UFI scatter ───────────────────────────────────────────────────

def plot_speed_vs_ufi(df: pd.DataFrame):
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(sample["avg_speed"], sample["ufi_score"],
                    c=sample["ufi_score"], cmap="RdYlGn_r",
                    alpha=0.5, s=18)
    plt.colorbar(sc, ax=ax, label="UFI Score")
    ax.set_xlabel("Average Speed (km/h)")
    ax.set_ylabel("UFI Score")
    ax.set_title("Speed vs UFI Score (1 000 random road-hours)")
    fig.tight_layout()
    path = OUTPUT_DIR / "06_speed_vs_ufi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 9. Box plots by class ─────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame):
    class_order = ["Free Flow", "Low", "Moderate", "High", "Severe"]
    class_order = [c for c in class_order if c in df["ufi_class"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    features  = [("avg_speed", "Speed (km/h)"), ("volume", "Volume (veh/hr)"),
                 ("C3", "Network Stress (C3)"), ("nlp_severity", "NLP Severity (C4)")]

    for ax, (feat, label) in zip(axes.flat, features):
        sns.boxplot(data=df, x="ufi_class", y=feat, order=class_order,
                    palette="RdYlGn_r", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by UFI Class")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Feature Distributions across UFI Classes", y=1.01, fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / "07_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame):
    print("\n══════════════════════════════════════")
    print("   UFI DATA ANALYSIS")
    print("══════════════════════════════════════")
    summary_stats(df)
    nbhd = neighbourhood_summary(df)
    nbhd.to_csv(OUTPUT_DIR / "neighbourhood_summary.csv")

    print("\nGenerating charts …")
    plot_hourly_ufi(df)
    plot_neighbourhood_bar(df)
    plot_component_heatmap(df)
    plot_correlation(df)
    plot_ufi_distribution(df)
    plot_speed_vs_ufi(df)
    plot_boxplots(df)
    print("\nAll analysis charts saved to:", OUTPUT_DIR)
    return nbhd


if __name__ == "__main__":
    df = pd.read_csv("data/ufi_scored.csv")
    run_analysis(df)
