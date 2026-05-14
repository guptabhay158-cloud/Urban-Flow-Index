"""
UFI Project — Master Pipeline
Runs the full pipeline end-to-end:
  Stage 1  →  Dataset Generation
  Stage 2  →  NLP Severity Scoring
  Stage 3  →  Graph Construction & Centrality
  Stage 4  →  UFI Score Computation
  Stage 5  →  Exploratory Data Analysis
  Stage 6  →  Predictive Modelling
  Stage 7  →  Tableau Export

Run:  python main.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_dataset   import build_dataset
from utils.nlp_severity       import score_dataframe
from models.graph_builder     import build_graph, compute_centrality, attach_centrality
from models.ufi_engine        import compute_ufi
from analysis.eda             import run_analysis
from models.modelling         import run_modelling
from tableau.tableau_export   import run_tableau_export


def banner(text: str):
    width = 50
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    t0 = time.time()

    # ── Stage 1: Generate Dataset ─────────────────────────────────────────────
    banner("Stage 1 / 7 — Dataset Generation")
    df = build_dataset()

    # ── Stage 2: NLP Severity Scoring ─────────────────────────────────────────
    banner("Stage 2 / 7 — NLP Severity Scoring")
    df = score_dataframe(df, text_col="incident_text")
    print(f"NLP scoring complete. Mean severity: {df['nlp_severity'].mean():.3f}")

    # ── Stage 3: Graph + Centrality ───────────────────────────────────────────
    banner("Stage 3 / 7 — Graph Construction & Betweenness Centrality")
    G          = build_graph(df)
    centrality = compute_centrality(G)
    df         = attach_centrality(df, centrality)
    print(f"Graph: {G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges")
    print(f"Centrality computed for {len(centrality)} roads")

    # ── Stage 4: UFI Computation ──────────────────────────────────────────────
    banner("Stage 4 / 7 — UFI Score Computation")
    df = compute_ufi(df)
    df.to_csv("data/ufi_scored.csv", index=False)
    print(f"UFI computed.  Mean score: {df['ufi_score'].mean():.2f}")
    print(f"Class distribution:\n{df['ufi_class'].value_counts().to_string()}")

    # ── Stage 5: EDA ──────────────────────────────────────────────────────────
    banner("Stage 5 / 7 — Exploratory Data Analysis")
    run_analysis(df)

    # ── Stage 6: Modelling ────────────────────────────────────────────────────
    banner("Stage 6 / 7 — Predictive Modelling")
    run_modelling(df)

    # ── Stage 7: Tableau Export ───────────────────────────────────────────────
    banner("Stage 7 / 7 — Tableau Export")
    run_tableau_export(df)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*50}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'═'*50}")
    print("\nKey output files:")
    print("  data/ufi_scored.csv            ← Full scored dataset")
    print("  analysis/01_hourly_ufi.png     ← City UFI by hour")
    print("  analysis/02_neighbourhood_bar  ← Peak-hour rankings")
    print("  analysis/03_component_heatmap  ← C1–C4 by neighbourhood")
    print("  analysis/08_reg_feature_imp.   ← Random Forest importance")
    print("  analysis/11_confusion_matrix   ← Classifier confusion matrix")
    print("  tableau/ufi_full.csv           ← Tableau-ready CSV")
    print("  tableau/UFI_Dashboard.twb      ← Tableau workbook (open in Desktop)")


if __name__ == "__main__":
    main()
