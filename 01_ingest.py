# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [markdown]
# # 01 – Ingest CNS Call/SMS Logs
# Converts raw CSV → prod-like Parquet partitioned by month.

# %%
import pandas as pd
from pathlib import Path

RAW = Path("data/cns_raw")
OUT = Path("parquet")
OUT.mkdir(exist_ok=True, parents=True)

def load_and_normalise() -> pd.DataFrame:
    calls = pd.read_csv(RAW / "calls" / "edges.csv")
    sms   = pd.read_csv(RAW / "sms" / "edges.csv")

    calls = calls.assign(channel="call")
    sms   = sms.assign(channel="sms")

    common = (
        pd.concat([calls, sms], ignore_index=True)
        .rename(columns={"user_a": "src", "user_b": "dst"})
    )

    # Partition by YYYY-MM for a prod-style layout
    common["yyyy_mm"] = pd.to_datetime(common["timestamp"]).dt.strftime("%Y-%m")
    for key, part in common.groupby("yyyy_mm"):
        part.drop(columns="yyyy_mm").to_parquet(OUT / f"year_month={key}.parquet", index=False)

    return common

if __name__ == "__main__":
    df = load_and_normalise()
    print("Rows written:", len(df))
