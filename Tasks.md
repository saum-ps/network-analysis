# Development Log — CNS Social-Graph POC

> Use this file to append dated, atomic notes as you work.  
> Reference the plan step in square brackets, e.g. `[1.2]`.

## Template

### YYYY-MM-DD  HH:MM
* **Progress**: _what you did_ `[step-id]`
* **Insight**: _something you learned / decided_
* **Next**: _immediate next micro-task_
* **Ref**: _commit hash / notebook cell / external source_

---

## Log
<!-- Start logging below this line -->

# Development Log — CNS Social-Graph POC

> Use this file to append dated, atomic notes as you work.  
> Reference the plan step in square brackets, e.g. `[1.2]`.

---

### 2025-08-05  10:10
* **Progress**: Created `.gitignore` and committed to Git `[0.3]`
* **Insight**: Ensure `.ipynb_checkpoints/`, `parquet/`, `.DS_Store`, and raw data paths are excluded.
* **Next**: Install Python dependencies and validate CUDA
* **Ref**: Commit `1`

---

### 2025-08-05  11:05
* **Progress**: Installed Conda, created virtual environment `socialgraph`, installed all required dependencies (pandas, pyarrow, jupytext, torch with CUDA support) `[0.1]`
* **Insight**: Verified CUDA is available via `torch.cuda.is_available()` and correct driver installed
* **Next**: Download and place CNS data
* **Ref**: Shell log, environment activated in Conda

---

### 2025-08-05  12:30
* **Progress**: Downloaded and extracted CNS call/sms data into `data/cns_raw/calls/` and `data/cns_raw/sms/` `[1.1]`
* **Insight**: CSVs are headerless; require manual column mapping during ingestion
* **Next**: Create ingestion script to convert to Parquet
* **Ref**: Local path `data/cns_raw/`

---

### 2025-08-05  13:00
* **Progress**: Created empty `parquet/` output folder `[1.3]`
* **Insight**: Keep structure clean, avoid raw CSVs in Git
* **Next**: Write `01_ingest.py`
* **Ref**: Local path `parquet/`

---

### 2025-08-05  14:15
* **Progress**: Drafted initial version of `01_ingest.py`, used placeholder logic with header assumption `[1.2]`
* **Insight**: Realized timestamp is interpreted incorrectly (assumed Unix but it’s offset from study start)
* **Next**: Fix column headers and timestamp logic
* **Ref**: Code draft before fix

---

### 2025-08-05  15:30
* **Progress**: Ran `jupytext --set-formats ipynb,py:percent 01_ingest.py` to enable notebook syncing `[0.3]`
* **Insight**: Jupytext config now embedded in script; ensures bi-directional sync
* **Next**: Fix errors in `01_ingest.py` and test
* **Ref**: Commit `2`

---

### 2025-08-05  16:10
* **Progress**: Activated Conda environment inside Cursor and tested script run `[0.1]`
* **Insight**: No errors when running directly from Cursor's terminal
* **Next**: Refactor and fix timestamp handling
* **Ref**: Cursor terminal log

---

### 2025-08-05  17:00
* **Progress**: Rewrote `01_ingest.py` to handle:
    - proper column mapping (`source`, `target`, `timestamp`, `duration`)
    - relative timestamp conversion using `datetime(2013, 9, 1)` as base
    - correct Parquet output `[1.2]`, `[1.3]`
* **Insight**: Based on CNS study design, timestamp is offset from study start
* **Next**: Check partitioned files, validate schema
* **Ref**: Commit `3`, `01_ingest.py` final version

---

### 2025-08-05  18:10
* **Progress**: Ran `jupytext --sync 01_ingest.py` to sync notebook with script `[0.3]`
* **Insight**: Now edits in either `.ipynb` or `.py` stay consistent
* **Next**: Begin implementing `build_multilayer_graph()` in `02_graph_build.ipynb`
* **Ref**: Commit `4`


---

### 2025-08-05  18:25
* **Progress**: Created `utils/device.py` to provide flexible GPU/CPU selector `[0.1]`
* **Insight**: Allows runtime override via `SOCIAL_DEVICE` env var (e.g., `"cpu"` or `"cuda:1"`); defaults to `cuda:0` if available
* **Next**: Use `get_device()` in graph-building or metric functions to place tensors on the correct device
* **Ref**: Commit `4`, file path: `utils/device.py`

---

### 2025-08-05  18:45
* **Progress**: Completed `SETUP_OK.txt` with CUDA verification `[0.2]`
* **Insight**: CUDA is available with NVIDIA GeForce GTX 1650; environment ready for GPU-accelerated graph processing
* **Next**: Begin Day 2 - create `02_graph_build.ipynb` and implement `build_multilayer_graph()`
* **Ref**: `SETUP_OK.txt` contains device details, Day 0 complete

---

### 2025-08-05  19:15
* **Progress**: Created `02_graph_build.py` and implemented core graph functions `[2.1]`, `[2.2]`
* **Insight**: Successfully built multilayer graph (568 nodes, 27933 edges) → collapsed to weighted graph (568 nodes, 2102 edges)
* **Next**: Implement remaining 7 metric functions (detect_circles, relationship_trend, etc.)
* **Ref**: `02_graph_build.py`, `graph.pkl`, `node_metrics.csv` created

---

### 2025-08-05  19:45
* **Progress**: Implemented all 10 metric functions `[2.3]` and completed Day 2 deliverables
* **Insight**: All functions ≤ 25 lines as required; metrics include extrovert_score, churn_rate, spam detection
* **Next**: Begin Day 3 - create `03_analytics.ipynb` for community detection and temporal analysis
* **Ref**: `02_graph_build.py` complete, `node_metrics.csv` updated with comprehensive metrics

---

### 2025-08-06  19:45
* **Progress**: Corrected the `02_graph_build.py` script and successfully executed `03_analytics.py`
* **Insight**: Fixed NetworkX compatibility issues and datetime parsing problems
* **Next**: Review all deliverables and prepare final project handoff
* **Ref**: All Day 3 deliverables completed successfully

---

### 2025-08-06  20:15
* **Progress**: Created and executed `03_analytics.py` for Day 3 deliverables `[3.1]`, `[3.2]`, `[3.3]`
* **Insight**: Fixed graph conversion issues and datetime parsing; successfully generated community detection and trend plots
* **Next**: Complete final deliverables - generate PDF insights deck and final validation
* **Ref**: `circles.csv`, `trend_plot.png` created, community detection working

