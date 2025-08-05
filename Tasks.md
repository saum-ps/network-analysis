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

### 2025-08-04  10:10
* **Progress**: Created `.gitignore` and committed to Git `[0.3]`
* **Insight**: Ensure `.ipynb_checkpoints/`, `parquet/`, `.DS_Store`, and raw data paths are excluded.
* **Next**: Install Python dependencies and validate CUDA
* **Ref**: Commit `1`

---

### 2025-08-04  11:05
* **Progress**: Installed Conda, created virtual environment `socialgraph`, installed all required dependencies (pandas, pyarrow, jupytext, torch with CUDA support) `[0.1]`
* **Insight**: Verified CUDA is available via `torch.cuda.is_available()` and correct driver installed
* **Next**: Download and place CNS data
* **Ref**: Shell log, environment activated in Conda

---

### 2025-08-04  12:30
* **Progress**: Downloaded and extracted CNS call/sms data into `data/cns_raw/calls/` and `data/cns_raw/sms/` `[1.1]`
* **Insight**: CSVs are headerless; require manual column mapping during ingestion
* **Next**: Create ingestion script to convert to Parquet
* **Ref**: Local path `data/cns_raw/`

---

### 2025-08-04  13:00
* **Progress**: Created empty `parquet/` output folder `[1.3]`
* **Insight**: Keep structure clean, avoid raw CSVs in Git
* **Next**: Write `01_ingest.py`
* **Ref**: Local path `parquet/`

---

### 2025-08-04  14:15
* **Progress**: Drafted initial version of `01_ingest.py`, used placeholder logic with header assumption `[1.2]`
* **Insight**: Realized timestamp is interpreted incorrectly (assumed Unix but it’s offset from study start)
* **Next**: Fix column headers and timestamp logic
* **Ref**: Code draft before fix

---

### 2025-08-04  15:30
* **Progress**: Ran `jupytext --set-formats ipynb,py:percent 01_ingest.py` to enable notebook syncing `[0.3]`
* **Insight**: Jupytext config now embedded in script; ensures bi-directional sync
* **Next**: Fix errors in `01_ingest.py` and test
* **Ref**: Commit `2`

---

### 2025-08-04  16:10
* **Progress**: Activated Conda environment inside Cursor and tested script run `[0.1]`
* **Insight**: No errors when running directly from Cursor's terminal
* **Next**: Refactor and fix timestamp handling
* **Ref**: Cursor terminal log

---

### 2025-08-04  17:00
* **Progress**: Rewrote `01_ingest.py` to handle:
    - proper column mapping (`source`, `target`, `timestamp`, `duration`)
    - relative timestamp conversion using `datetime(2013, 9, 1)` as base
    - correct Parquet output `[1.2]`, `[1.3]`
* **Insight**: Based on CNS study design, timestamp is offset from study start
* **Next**: Check partitioned files, validate schema
* **Ref**: Commit `3`, `01_ingest.py` final version

---

### 2025-08-04  18:10
* **Progress**: Ran `jupytext --sync 01_ingest.py` to sync notebook with script `[0.3]`
* **Insight**: Now edits in either `.ipynb` or `.py` stay consistent
* **Next**: Begin implementing `build_multilayer_graph()` in `02_graph_build.ipynb`
* **Ref**: Commit `4`


---

### 2025-08-04  18:25
* **Progress**: Created `utils/device.py` to provide flexible GPU/CPU selector `[0.1]`
* **Insight**: Allows runtime override via `SOCIAL_DEVICE` env var (e.g., `"cpu"` or `"cuda:1"`); defaults to `cuda:0` if available
* **Next**: Use `get_device()` in graph-building or metric functions to place tensors on the correct device
* **Ref**: Commit `4`, file path: `utils/device.py`