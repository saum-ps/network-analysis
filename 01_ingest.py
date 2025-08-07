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
from datetime import datetime, timedelta

RAW = Path("data/cns_raw")
OUT = Path("parquet")
OUT.mkdir(exist_ok=True, parents=True)

def load_and_normalise() -> pd.DataFrame:
    # Read CSV files with proper column names
    calls = pd.read_csv(RAW / "calls" / "edges.csv", 
                       names=["source", "target", "timestamp", "duration"],
                       comment="#")
    sms   = pd.read_csv(RAW / "sms" / "edges.csv", 
                       names=["source", "target", "timestamp"],
                       comment="#")

    calls = calls.assign(channel="call")
    sms   = sms.assign(channel="sms")

    common = (
        pd.concat([calls, sms], ignore_index=True)
        .rename(columns={"source": "src", "target": "dst"})
    )

    # Convert relative timestamps to absolute dates
    # Based on CNS documentation: timestamps are seconds from study start
    # Study started on a Sunday during school term (likely 2013)
    # For now, using a reasonable reference date - adjust as needed
    study_start = datetime(2013, 9, 1)  # Approximate study start date
    
    # Convert relative seconds to absolute datetime
    common["sent"] = study_start + pd.to_timedelta(common["timestamp"], unit='s')
    common["yyyy_mm"] = common["sent"].dt.strftime("%Y-%m")
    
    for key, part in common.groupby("yyyy_mm"):
        part.drop(columns=["yyyy_mm"]).to_parquet(OUT / f"year_month={key}.parquet", index=False)

    return common

if __name__ == "__main__":
    df = load_and_normalise()
    print("Rows written:", len(df))
    print("Date range:", df["sent"].min(), "to", df["sent"].max())
