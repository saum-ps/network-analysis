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
MIN_TREND_WEIGHT: int = 2 * SMS_WEIGHT  # minimal recent activity to make a call
TREND_WIN_DAYS: int = 7       # rolling‑window size for trend/churn
MIN_EVENTS_PREF: int = 5      # min interactions to label channel preference
PREF_THRESHOLD: float = 0.60  # 60 % rule
MIN_WEIGHT_TOPTIE: int = 90   # ignore ties < 90 s total weight

REPLY_WINDOW: str = "1H"      # only count replies within this window
SESSION_GAP: str = "2H"       # break conversation after this gap
MIN_REPLY_PAIRS: int = 3      # need at least this many reply pairs
REPLY_USE_MEDIAN: bool = True # use median over mean for robustness
BET_SCALE: int = 1000         # keep the betweenness centrality scale

MIN_WEEKLY_BASE: int = 3 * SMS_WEIGHT   # need at least this inbound last week to judge a drop
MIN_RECENT_SUM: int  = 5 * SMS_WEIGHT   # min (prev + curr) to avoid tiny-sample noise
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
        G.add_edge(row.src, row.dst, key=f"{row.channel}_{row.timestamp}", channel=row.channel, weight=weight, sent=row.sent)
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
                "missed_count": 0,
                "timestamps": []
            }
            cg.add_edge(u, v, **edge_data)
        
        # Record timestamp & weight for churn analysis
        if "sent" in data:
            edge_data["timestamps"].append((data["sent"], data["weight"]))
        
        if data["channel"] == "call":
            if data["weight"] < 0:
                edge_data["missed_count"] += 1
            else:
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


# replace the whole relationship_trend() with:
def relationship_trend(df_pair: pd.DataFrame) -> str:
    if df_pair.empty or "sent" not in df_pair or "weight" not in df_pair:
        return "insufficient_data"
    weekly = (
        df_pair.set_index("sent")
               .sort_index()["weight"]
               .resample(f"{TREND_WIN_DAYS}D").sum()
               .fillna(0.0)
    )
    if len(weekly) < 3:
        return "insufficient_data"
    prev, curr = weekly.iloc[-2], weekly.iloc[-1]
    # require some minimum signal to avoid labeling pure noise
    if (prev + curr) < MIN_TREND_WEIGHT:
        return "insufficient_data"
    if prev == 0:
        return "growing" if curr >= MIN_TREND_WEIGHT else "insufficient_data"
    change = (curr - prev) / prev
    if change > 0.25:
        return "growing"
    if change < -0.25:
        return "fading"
    return "stable"



def avg_reply_delay(
    df_pair: pd.DataFrame,
    reply_window: str | None = None,
    session_gap: str | None = None,
    min_pairs: int | None = None,
    use_median: bool | None = None
) -> float | None:
    import numpy as np
    if df_pair.empty:
        return None

    # SMS-only filter (if channel present)
    if "channel" in df_pair.columns:
        df = df_pair[df_pair["channel"] == "sms"].copy()
        if df.empty:
            return None
    else:
        df = df_pair.copy()

    df = df.sort_values("sent")[["src", "dst", "sent"]].rename(columns={"src": "sender"})

    # Resolve params from constants
    rw = pd.Timedelta(reply_window or REPLY_WINDOW)
    sg = pd.Timedelta(session_gap or SESSION_GAP)
    min_pairs = int(min_pairs or MIN_REPLY_PAIRS)
    use_median = REPLY_USE_MEDIAN if use_median is None else use_median

    # Burst compression + session breaks
    gap = df["sent"].diff()
    new_run = (df["sender"] != df["sender"].shift(1)) | (gap > sg)
    runs = df.loc[new_run, ["sender", "sent"]].reset_index(drop=True)

    # Replies = alternations within reply window, measure from FIRST msg of prior burst
    delays = []
    for i in range(1, len(runs)):
        if runs.at[i, "sender"] != runs.at[i-1, "sender"]:
            dt = (runs.at[i, "sent"] - runs.at[i-1, "sent"]).total_seconds()
            if 0 <= dt <= rw.total_seconds():
                delays.append(dt)

    if len(delays) < min_pairs:
        return None
    return float(np.median(delays) if use_median else sum(delays) / len(delays))



def extrovert_score(G: nx.DiGraph, top_n: int = 10) -> List[Tuple[int, float]]:
    # total interaction strength (in + out)
    strength = {
        n: sum(d["weight"] for _,_,d in G.out_edges(n, data=True)) +
           sum(d["weight"] for _,_,d in G.in_edges(n, data=True))
        for n in G
    }
    # unweighted betweenness so strong ties aren't treated as long distances
    bet = nx.betweenness_centrality(G, weight=None, normalized=True)
    score = {n: strength.get(n, 0) + bet.get(n, 0) * BET_SCALE for n in G}
    return sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]


def churn_drop(G: nx.DiGraph, user: int) -> Tuple[bool, float]:
    inbound = [data["weight"] for _, _, data in G.in_edges(user, data=True)]
    if not inbound:
        return False, 0.0

    rows = []
    for src, _, data in G.in_edges(user, data=True):
        rows.extend([(src, t, w) for t, w in data.get("timestamps", [])])
    if not rows:
        return False, 0.0

    df = (pd.DataFrame(rows, columns=["src", "sent", "weight"])
            .set_index("sent").sort_index())
    weekly = df["weight"].resample("7D").sum().fillna(0.0)
    if len(weekly) < 2:
        return False, 0.0

    prev, curr = float(weekly.iloc[-2]), float(weekly.iloc[-1])
    ratio = curr / max(1e-6, prev)

    # noise guards: don’t flag churn on tiny activity
    if prev < MIN_WEEKLY_BASE or (prev + curr) < MIN_RECENT_SUM:
        return False, ratio

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
