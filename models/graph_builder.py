"""
UFI Project — Graph Builder
Constructs a directed road network G = (V, E) using NetworkX,
computes node and edge betweenness centrality, and returns
a centrality lookup that feeds into the UFI formula as C3 (Network Stress).
"""

import random
import numpy as np
import pandas as pd
import networkx as nx

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph from the road dataset.

    Nodes  = unique intersections (inferred from road names)
    Edges  = road segments connecting two intersections
    Weight = inverse of average_speed  (lower speed → higher travel cost)

    Parameters
    ----------
    df : DataFrame
        Raw UFI dataset (output of generate_dataset.py).
        Expected columns: road_id, road_name, neighbourhood, avg_speed, volume.

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # One node per unique road endpoint within each neighbourhood
    # We simulate intersections by treating each road as connecting two nodes
    road_meta = df.groupby("road_id").agg(
        road_name=("road_name", "first"),
        neighbourhood=("neighbourhood", "first"),
        avg_speed=("avg_speed", "mean"),
        volume=("volume", "mean"),
        capacity=("capacity", "first"),
    ).reset_index()

    # Add intersection nodes (one "in" and one "out" node per road for directed graph)
    for _, row in road_meta.iterrows():
        u = f"{row['road_id']}_in"
        v = f"{row['road_id']}_out"
        G.add_node(u, neighbourhood=row["neighbourhood"])
        G.add_node(v, neighbourhood=row["neighbourhood"])

        travel_cost = 1 / max(row["avg_speed"], 1)   # avoid div/0
        G.add_edge(
            u, v,
            road_id=row["road_id"],
            road_name=row["road_name"],
            neighbourhood=row["neighbourhood"],
            avg_speed=row["avg_speed"],
            volume=row["volume"],
            capacity=row["capacity"],
            weight=travel_cost,
        )

    # Connect roads within the same neighbourhood (simulate junctions)
    roads_by_nbhd = road_meta.groupby("neighbourhood")["road_id"].apply(list)
    for nbhd, roads in roads_by_nbhd.items():
        random.shuffle(roads)
        for i in range(len(roads) - 1):
            u = f"{roads[i]}_out"
            v = f"{roads[i+1]}_in"
            # Connector edge — use mean speed of both roads
            avg = (road_meta.loc[road_meta.road_id == roads[i], "avg_speed"].values[0] +
                   road_meta.loc[road_meta.road_id == roads[i+1], "avg_speed"].values[0]) / 2
            G.add_edge(u, v, weight=1 / max(avg, 1), road_id=None)

    # Cross-neighbourhood connections (arterials)
    nbhds = list(roads_by_nbhd.keys())
    for i in range(len(nbhds) - 1):
        r1 = random.choice(roads_by_nbhd[nbhds[i]])
        r2 = random.choice(roads_by_nbhd[nbhds[i + 1]])
        G.add_edge(f"{r1}_out", f"{r2}_in", weight=0.05, road_id=None)

    return G


def compute_centrality(G: nx.DiGraph) -> dict:
    """
    Compute normalised edge betweenness centrality.

    Returns
    -------
    dict : {road_id → centrality_score (0–1)}
    """
    print("Computing edge betweenness centrality (may take ~10 s) …")
    ebc = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)

    # Aggregate edge centrality per road_id
    road_centrality: dict[str, list] = {}
    for (u, v), score in ebc.items():
        road_id = G[u][v].get("road_id")
        if road_id:
            road_centrality.setdefault(road_id, []).append(score)

    # Average over all edges belonging to the same road
    return {rid: float(np.mean(scores)) for rid, scores in road_centrality.items()}


def attach_centrality(df: pd.DataFrame, centrality: dict) -> pd.DataFrame:
    """
    Merge centrality scores into the main DataFrame as column ``network_stress``.
    Missing roads (junction-only edges) receive the median score.
    """
    df = df.copy()
    median_score = float(np.median(list(centrality.values()))) if centrality else 0.0
    df["network_stress"] = df["road_id"].map(centrality).fillna(median_score)
    return df


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.generate_dataset import build_dataset

    df = build_dataset()
    G  = build_graph(df)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    centrality = compute_centrality(G)
    print(f"Centrality computed for {len(centrality)} roads")
    print("Top 5 most critical roads:")
    for rid, score in sorted(centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"  {rid}  →  {score:.4f}")
