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

# %%
"""
03 – Day‑3 Analytics & Slide Export (CNS POC)
------------------------------------------------
Generates final artefacts:
* `circles.csv` – community label per user
* Growth/fade/churn examples & two matplotlib plots
* `day3_insights.pdf` – 3‑slide exec deck (via python‑pptx)

Assumes `graph.pkl` and Parquet ingest exist.
"""

# %% [imports]
from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# %% [paths & constants]
GRAPH_PKL = Path("graph.pkl")
PARQUET_DIR = Path("parquet")
CIRCLES_CSV = Path("circles.csv")
SLIDE_DECK = Path("day3_insights.pdf")  # python‑pptx saves pptx; we’ll convert to pdf via libreoffice if available

# %% [load data]
print("Loading collapsed graph …")
with GRAPH_PKL.open("rb") as f:
    G: nx.DiGraph = pickle.load(f)

df_events = pd.concat(pd.read_parquet(p) for p in PARQUET_DIR.rglob("*.parquet"))

# ---------------------------------------------------------------------------
# 1  Community detection & circles.csv
# ---------------------------------------------------------------------------
print("Running community detection …")
UG = nx.Graph()
for u, v, d in G.edges(data=True):
    weight = d.get("weight", 0)
    if weight > 0:  # Only add edges with positive weights
        UG.add_edge(u, v, weight=weight)
import networkx.algorithms.community as nx_comm
# Use Leiden /Louvain (pick Louvain for simplicity)
communities = nx_comm.louvain_communities(UG, weight="weight")

user2circle = {}
for cid, comm in enumerate(communities):
    for node in comm:
        user2circle[node] = cid

pd.Series(user2circle, name="circle_id").to_csv(CIRCLES_CSV, header=True)
print("Wrote", CIRCLES_CSV)

# ---------------------------------------------------------------------------
# 2  Build temporal edge series for trend & churn demo plots
# ---------------------------------------------------------------------------
print("Computing weekly edge weights …")
# Create a YYYY‑MM‑DD column for grouping
if "sent" not in df_events.columns:
    df_events["sent"] = pd.to_datetime(df_events["timestamp"], unit="s", origin="2013-09-01")

df_events["week"] = df_events["sent"].dt.to_period("W-SUN")

df_pair_week = (
    df_events.assign(weight=lambda d: (d["duration"].fillna(0) + (d["channel"] == "sms") * 30))
    .groupby(["src", "dst", "week"], as_index=False)["weight"].sum()
)

# Pick top 2 strongest pairs for illustrative trend lines
pair_totals = df_pair_week.groupby(["src", "dst"])["weight"].sum().nlargest(2).index

plt.figure()
for src, dst in pair_totals:
    serie = (
        df_pair_week.query("src==@src and dst==@dst")
        .set_index("week")["weight"]
        .asfreq("W-SUN", fill_value=0)
        .sort_index()
    )
    serie.plot(label=f"{src}→{dst}")
plt.title("Weekly Edge Weight (Top 2 Pairs)")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()
plt.savefig("trend_plot.png")
print("Saved trend_plot.png")