# 3-Day Social-Graph POC — Development Plan

## 1 . Problem Statement  
Our photo-cloud platform needs a lightweight *social-graph* service that reveals  
*who* matters to each user, *how strongly* they are connected, and *how those ties change over time*.  
Because production data are unavailable, we will prototype on the **Copenhagen Networks Study** (CNS) 4-week call/SMS subset.

## 2 . Objective  
Deliver in **3 days** a GPU-runnable proof-of-concept that:  

| Day | Milestone | Key Artefacts | Why It Matters |
|----|-----------|--------------|----------------|
| 0   | Environment check | `SETUP_OK.txt` | Confirms CUDA ready |
| 1   | Data ingest & parquet | `01_ingest.ipynb`, `parquet/` | Reproducible ETL mirroring prod schema |
| 2   | Graph & metrics | `02_graph_build.ipynb`, `graph.pkl`, `node_metrics.csv` | Core multi-layer graph & tie-strength metrics |
| 3   | Circles & trends | `03_analytics.ipynb`, `circles.csv`, `day3_insights.pdf` | Community detection, trend flags, slide deck |

Meeting **all artefacts** each day is mandatory.

## 3 . Dataset in Scope  
* `sms.csv.zip` – timestamp, sender, recipient  
* `calls.csv.zip` – timestamp, caller, callee, duration  
These map 1-to-1 onto our internal schema and cover ~700 users.

## 4 . Constraints & Assumptions  
* **Runtime**: each notebook must re-run in < 120 s on GPU VM.  
* **Edge weight**: `weight = duration_sec + 30 * sms_count`.  
* **Coding standards**: black-formatted, typed, doc-stringed.  
* **No raw CSV/ZIP in git** — only Parquet or pickles (git-ignored).  
* Must allow **GPU selector** → default to local GTX 1650, but overridable via `SOCIAL_DEVICE` env var (used by `utils/device.py`)


## 5 . Insights & Design Notes  
* Treat SMS and CALL as **layers** in a `networkx.MultiDiGraph`.  
* Collapse to weighted **directed** graph for metrics.  
* Ten callable functions (≤ 25 lines each) answer the scorecard questions (top ties, reciprocity, churn, spam, …).  
* Temporal signals computed on **7-day rolling windows** (trend, churn, bursts).

*CNS timestamps are relative to study start, not Unix. We assume study_start = datetime(2013, 9, 1) based on metadata from the CNS paper and weekly activity patterns.
* GPU/CPU device handling is centralized in `utils/device.py` → used in any Torch-based logic; supports override via `SOCIAL_DEVICE` env var.



## 6 . Atomic Development Steps
> Each task number is referenced in parentheses for `TASKS.md` to link back.

### Day 0 – Setup
0.1  Create Conda env `socialgraph`; install deps (pandas, pyarrow, networkx, torch + CUDA)  
0.2  Verify `torch.cuda.is_available()`; write `SETUP_OK.txt` with selected GPU name  
0.3  Stub project repo: `data/ raw/ notebooks/ src/ docs/`

### Day 1 – Ingest & Schema
1.1  Download and unzip CNS `sms.csv.zip`, `calls.csv.zip`  
1.2  Implement `load_and_normalise()` → canonical DF [`id`, `type`, `sender`, `recipient`, `sent`, `duration`, `direction`]  

*Implement load_and_normalise():
    -Read raw CSVs without headers
    -Use column names: ["source", "target", "timestamp", "duration"] (calls), and skip duration for SMS
    -Convert timestamp → datetime using relative offset
    -Return canonical DF: ["src", "dst", "timestamp", "duration", "channel"]
1.3  Concatenate layers; partition-write Parquet `parquet/type=.../year=YYYY/month=MM/`  
1.4  Unit tests: row counts, null checks, schema match

### Day 2 – Graph & Metrics
2.1  `build_multilayer_graph(df)` → `nx.MultiDiGraph` with edge attrs layer/duration/sent  
2.2  `collapse_edges(G)` → weighted `nx.DiGraph` (edge weight formula)  
2.3  Implement 10 metric callables (scorecard) in ≤ 25 lines each  

| #  | Callable                                   | Business question                           | Diff / Pts |
| -- | ------------------------------------------ | ------------------------------------------- | ---------- |
| 1  | `top_ties`                                 | User’s top-five ties                        | Easy / 10  |
| 2  | `channel_preference`                       | Preferred channel (text vs call) for a pair | Easy / 10  |
| 3  | `reciprocity`                              | User-level reciprocity score                | Easy / 10  |
| 4  | `detect_circles`                           | Discover social circles **per** user        | Med / 15   |
| 5  | `relationship_trend`                       | Is contact Z growing or fading?             | Med / 15   |
| 6  | ~~`new_contact_bursts`~~ *(skip this one)* | First-week burst of new contacts            | —          |
| 7  | `avg_reply_delay`                          | Average reply delay between two users       | Med / 15   |
| 8  | `extrovert_score`                          | Biggest extroverts / bridges in the graph   | Hard / 25  |
| 9  | `churn_drop`                               | Weekly inbound-drop churn detector          | Hard / 25  |
| 10 | `find_spam_nodes`                          | Spam / harassment node finder               | Hard / 25  |


2.4  Export `graph.pkl`, `node_metrics.csv`; smoke-test metrics

### Day 3 – Circles & Temporal Analytics
3.1  Community detection on undirected weighted graph → `detect_circles()`  
3.2  Temporal trend functions (`relationship_trend`, `churn_drop`, `new_contact_bursts`, etc.)  
3.3  Generate two illustrative plots (degree vs time, community size dist.)  
3.4  Summarise findings in ≤ 3-slide PDF (`day3_insights.pdf`)  

### Wrap-up
4.1  Ensure notebooks run top-to-bottom < 120 s on GPU  
4.2  Tag git commit, push artefacts, hand-off

## 7 . Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU compat / driver issues | High | Allow CPU fallback; provide env selector |
| Large graph memory | Med | Use `networkx` then export to `igraph` if > 1 M edges |
| Time-boxed deliverables | High | Maintain daily checkpoints; prune non-essential visuals |
| Challenge functions exceed 25 lines | Low | Refactor helpers, use vectorised ops |

## 8 . Deliverables Checklist
- [ ] `SETUP_OK.txt`  
- [ ] `01_ingest.ipynb` + Parquet folder  
- [ ] `02_graph_build.ipynb`, `graph.pkl`, `node_metrics.csv`  
- [ ] `03_analytics.ipynb`, `circles.csv`, `day3_insights.pdf`  
- [ ] Code formatted & typed; all tests pass
