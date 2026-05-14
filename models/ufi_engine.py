"""
UFI Project — Core UFI Engine
Computes the Urban Flow Index for every road-hour record.

Formula (from project document):
    UFI = (0.35×C1 + 0.30×C2 + 0.20×C3 + 0.15×C4) × TemporalWeight × 100

Components
----------
C1  Link Performance    1 − (avg_speed / speed_limit)
C2  Volume/Capacity     volume / capacity
C3  Network Stress      edge betweenness centrality (from graph module)
C4  NLP Severity        rule-based keyword score   (from NLP module)

All four components are Min-Max normalised before combining.
TemporalWeight: peak = 1.0, shoulder = 0.7, night = 0.2

Classification bands
--------------------
0–15   Free Flow
15–35  Low
35–55  Moderate
55–75  High
75–100 Severe
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Temporal weights ──────────────────────────────────────────────────────────

def temporal_weight(hour: int) -> float:
    """Map hour (0–23) → temporal weight."""
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 1.0      # morning / evening peak
    elif 12 <= hour <= 14:
        return 0.7      # midday shoulder
    elif 21 <= hour <= 23 or 0 <= hour <= 5:
        return 0.2      # night
    else:
        return 0.5      # off-peak


# ── Classification ────────────────────────────────────────────────────────────

BANDS = [
    (0,  15,  "Free Flow"),
    (15, 35,  "Low"),
    (35, 55,  "Moderate"),
    (55, 75,  "High"),
    (75, 100, "Severe"),
]

def classify_ufi(score: float) -> str:
    for lo, hi, label in BANDS:
        if lo <= score < hi:
            return label
    return "Severe"   # exactly 100


# ── Min-Max normaliser (per-column, fit on full dataset) ──────────────────────

def minmax_normalise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with specified columns scaled to [0, 1]."""
    df = df.copy()
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


# ── Component computation ─────────────────────────────────────────────────────

def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw component columns C1–C4 to the DataFrame.

    Requires columns: avg_speed, speed_limit, volume, capacity,
                      network_stress, nlp_severity
    """
    df = df.copy()
    df["C1_raw"] = 1 - (df["avg_speed"] / df["speed_limit"])   # link performance
    df["C1_raw"] = df["C1_raw"].clip(0, 1)

    df["C2_raw"] = (df["volume"] / df["capacity"]).clip(0, 1)  # volume/capacity

    df["C3_raw"] = df["network_stress"]                         # already [0,1]

    df["C4_raw"] = df["nlp_severity"]                          # already [0,1]

    return df


def compute_ufi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full UFI computation pipeline.

    1. Compute raw components
    2. Min-Max normalise each component across the whole dataset
    3. Apply weighted sum
    4. Apply temporal weight
    5. Scale to 0–100
    6. Classify

    Parameters
    ----------
    df : DataFrame with columns produced by earlier pipeline stages.

    Returns
    -------
    DataFrame with added columns: C1–C4, ufi_raw, temporal_weight, ufi_score, ufi_class
    """
    df = compute_components(df)

    # Normalise raw components
    df = minmax_normalise(df, ["C1_raw", "C2_raw", "C3_raw", "C4_raw"])
    df.rename(columns={
        "C1_raw": "C1", "C2_raw": "C2", "C3_raw": "C3", "C4_raw": "C4"
    }, inplace=True)

    # Weighted composite
    df["ufi_raw"] = (
        0.35 * df["C1"] +
        0.30 * df["C2"] +
        0.20 * df["C3"] +
        0.15 * df["C4"]
    )

    # Temporal weight
    df["temporal_weight"] = df["hour"].apply(temporal_weight)

    # Final score (0–100)
    df["ufi_score"] = (df["ufi_raw"] * df["temporal_weight"] * 100).clip(0, 100)

    # Classification
    df["ufi_class"] = df["ufi_score"].apply(classify_ufi)

    return df


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.generate_dataset import build_dataset
    from utils.nlp_severity   import score_dataframe
    from models.graph_builder  import build_graph, compute_centrality, attach_centrality

    df          = build_dataset()
    df          = score_dataframe(df)
    G           = build_graph(df)
    centrality  = compute_centrality(G)
    df          = attach_centrality(df, centrality)
    df          = compute_ufi(df)

    df.to_csv("data/ufi_scored.csv", index=False)
    print(f"\nSaved ufi_scored.csv  ({len(df)} rows)")
    print("\nNeighbourhood peak-hour UFI (mean):")
    peak = df[df["temporal_weight"] == 1.0].groupby("neighbourhood")["ufi_score"].mean().sort_values(ascending=False)
    print(peak.round(2).to_string())
