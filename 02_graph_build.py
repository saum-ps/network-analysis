# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Day 2: Graph Building & Core Metrics
#
# **Objective**: Build multilayer social graph from CNS data and implement core tie-strength metrics.
#
# **Deliverables**:
# - `build_multilayer_graph(df)` → `nx.MultiDiGraph`
# - `collapse_edges(G)` → weighted `nx.DiGraph`
# - 10 metric callables (≤ 25 lines each)
# - Export `graph.pkl`, `node_metrics.csv`

# %%
# %%
# %%
"""
02 – Build Social Graph & Metrics (CNS POC)
------------------------------------------------
Reads all Parquet partitions, constructs a multilayer graph, collapses it
into a single weighted digraph, and exposes the nine analytics callables.
"""

# %% [imports]
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
from datetime import timedelta
from utils.device import get_device  # GPU helper (currently unused but kept for future GNN work)

# %% [constants]
SMS_WEIGHT: int = 30          # seconds‑credit per SMS
TREND_WIN_DAYS: int = 7       # rolling‑window size for trend/churn
MIN_EVENTS_PREF: int = 5      # min interactions to label channel preference
PREF_THRESHOLD: float = 0.60  # 60 % rule
MIN_WEIGHT_TOPTIE: int = 90   # ignore ties < 90 s total weight
CHURN_DROP_RATIO: float = 0.40  # current wk inbound < 40 % prev wk

RAW_PARQUET_DIR = Path("parquet")
GRAPH_PKL = Path("graph.pkl")
NODE_CSV = Path("node_metrics.csv")

# %% [loader]
def load_all_partitions(parquet_dir: Path = RAW_PARQUET_DIR) -> pd.DataFrame:
    """Load every Parquet file under *parquet_dir* into one DataFrame."""
    parts = list(parquet_dir.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError("No Parquet files found – run 01_ingest first.")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)

# %% [graph builders]
def build_multilayer_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Return MultiDiGraph with separate edges for call vs sms (directed)."""
    G = nx.MultiDiGraph()
    for row in df.itertuples(index=False):
        weight = row.duration if row.channel == "call" else SMS_WEIGHT
        key = row.channel  # distinguishes parallel edges
        G.add_edge(row.src, row.dst, key=key, channel=row.channel, weight=weight)
    return G


def collapse_edges(mg: nx.MultiDiGraph) -> nx.DiGraph:
    """Sum weights per direction; keep channel‑specific counts in attributes."""
    cg = nx.DiGraph()
    for u, v, data in mg.edges(data=True):
        # Get existing edge data or create new
        if cg.has_edge(u, v):
            edge_data = cg[u][v]
        else:
            edge_data = {
                "weight": 0,
                "call_weight": 0,
                "sms_weight": 0,
                "call_count": 0,
                "sms_count": 0,
            }
            cg.add_edge(u, v, **edge_data)
        
        if data["channel"] == "call":
            edge_data["call_weight"] += data["weight"]
            edge_data["call_count"] += 1
        else:
            edge_data["sms_weight"] += data["weight"]
            edge_data["sms_count"] += 1
        edge_data["weight"] = edge_data["call_weight"] + edge_data["sms_weight"]
    return cg

# %% [callables]

def top_ties(G: nx.DiGraph, user: int, k: int = 5, min_weight: int = MIN_WEIGHT_TOPTIE) -> List[Tuple[int, float]]:
    neigh: Dict[int, float] = {}
    for nbr in G.successors(user):
        neigh[nbr] = neigh.get(nbr, 0) + G[user][nbr]["weight"]
    for nbr in G.predecessors(user):
        neigh[nbr] = neigh.get(nbr, 0) + G[nbr][user]["weight"]
    return sorted(
        [(n, w) for n, w in neigh.items() if w >= min_weight],
        key=lambda x: x[1],
        reverse=True,
    )[:k]


def channel_preference(G: nx.DiGraph, a: int, b: int) -> str:
    edge_ab = G.get_edge_data(a, b, default={})
    edge_ba = G.get_edge_data(b, a, default={})
    sms = edge_ab.get("sms_count", 0) + edge_ba.get("sms_count", 0)
    calls = edge_ab.get("call_count", 0) + edge_ba.get("call_count", 0)
    total = sms + calls
    if total < MIN_EVENTS_PREF:
        return "undetermined"
    sms_ratio = sms / total
    if sms_ratio >= PREF_THRESHOLD:
        return "sms"
    if sms_ratio <= 1 - PREF_THRESHOLD:
        return "call"
    return "balanced"


def reciprocity(G: nx.DiGraph, user: int) -> float:
    in_w = sum(data["weight"] for _, _, data in G.in_edges(user, data=True))
    total_w = in_w + sum(data["weight"] for _, _, data in G.out_edges(user, data=True))
    return 0.0 if total_w == 0 else in_w / total_w


def detect_circles(G: nx.DiGraph, user: int) -> List[int]:
    import networkx.algorithms.community as nx_comm

    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        UG.add_edge(u, v, weight=data["weight"])
    for res in [1.0, 0.5, 1.5]:  # retry with diff resolution if degenerate
        part = nx_comm.louvain_communities(UG, weight="weight", resolution=res)
        if 1 < len(part) < len(UG):
            break
    for cid, community in enumerate(part):
        if user in community:
            return list(community)
    return []


def relationship_trend(df_pair: pd.DataFrame) -> str:
    if df_pair.empty:
        return "insufficient_data"
    df_pair = df_pair.set_index("sent").sort_index()
    weekly = df_pair["weight"].resample(f"{TREND_WIN_DAYS}D").sum()
    if len(weekly) < 3:
        return "insufficient_data"
    change = (weekly.iloc[-1] - weekly.iloc[-2]) / max(1e-6, weekly.iloc[-2])
    if change > 0.25:
        return "growing"
    if change < -0.25:
        return "fading"
    return "stable"


def avg_reply_delay(df_pair: pd.DataFrame) -> float | None:
    if df_pair.empty:
        return None
    df_pair = df_pair.sort_values("sent")
    delays = []
    last_out_ts = None
    last_out_sender = None
    for row in df_pair.itertuples(index=False):
        s, d, ts = row.src, row.dst, row.sent
        if last_out_ts is not None and last_out_sender != s:
            delays.append((ts - last_out_ts).total_seconds())
        last_out_ts, last_out_sender = ts, s
    return None if not delays else sum(delays) / len(delays)


def extrovert_score(G: nx.DiGraph, top_n: int = 10) -> List[Tuple[int, float]]:
    deg = {n: sum(data["weight"] for _, _, data in G.out_edges(n, data=True)) for n in G}
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    score = {n: deg.get(n, 0) + bet.get(n, 0) * 1000 for n in G}
    return sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]


def churn_drop(G: nx.DiGraph, user: int) -> Tuple[bool, float]:
    inbound = [data["weight"] for _, _, data in G.in_edges(user, data=True)]
    if not inbound:
        return False, 0.0
    # Build per‑edge DataFrame of timestamps & weights
    rows = []
    for src, _, data in G.in_edges(user, data=True):
        rows.extend([(src, t, w) for t, w in data.get("timestamps", [])])
    if not rows:
        return False, 0.0
    df = pd.DataFrame(rows, columns=["src", "sent", "weight"]).set_index("sent").sort_index()
    weekly = df["weight"].resample("7D").sum()
    if len(weekly) < 2:
        return False, 0.0
    ratio = weekly.iloc[-1] / max(1e-6, weekly.iloc[-2])
    return ratio < CHURN_DROP_RATIO, ratio


def find_spam_nodes(G: nx.DiGraph) -> List[int]:
    suspects = []
    for n in G.nodes:
        out_w = sum(data["weight"] for _, _, data in G.out_edges(n, data=True))
        in_w = sum(data["weight"] for _, _, data in G.in_edges(n, data=True))
        if out_w >= 20 * SMS_WEIGHT and (in_w == 0 or out_w / max(1, in_w) > 10):
            if reciprocity(G, n) < 0.1:
                suspects.append(n)
    return suspects

# %% [driver]
if __name__ == "__main__":
    df_full = load_all_partitions()

    # Build graphs
    mg = build_multilayer_graph(df_full)
    cg = collapse_edges(mg)

    # Save graph for Day‑3 use
    with GRAPH_PKL.open("wb") as f:
        pickle.dump(cg, f)

    # Produce node‑level metrics CSV
    rows = []
    for n in cg.nodes:
        rows.append({
            "node": n,
            "reciprocity": reciprocity(cg, n),
            "out_weight": sum(d["weight"] for _, _, d in cg.out_edges(n, data=True)),
            "in_weight": sum(d["weight"] for _, _, d in cg.in_edges(n, data=True)),
        })
    pd.DataFrame(rows).to_csv(NODE_CSV, index=False)

    print("Graph construction complete →", GRAPH_PKL)
    print("Node metrics written →", NODE_CSV)
