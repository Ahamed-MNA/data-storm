# 🌩️ DataStorm Round 2 — Team Data Mavericks

> **End-to-end ML pipeline** for outlet-level sales potential prediction, marketing spend optimisation, and Explainable AI narrative generation.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Data Layout](#data-layout)
7. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
   - [Stage 1 — Bronze Ingestion](#stage-1--bronze-ingestion)
   - [Stage 2 — Silver Cleaning](#stage-2--silver-cleaning)
   - [Stage 3 — POI & Competitor Graph](#stage-3--poi--competitor-graph)
   - [Stage 4 — Gold Feature Engineering](#stage-4--gold-feature-engineering)
   - [Stage 5 — SFA Model Training](#stage-5--sfa-model-training)
   - [Stage 6 — Predictions](#stage-6--predictions)
   - [Stage 7 — Marketing Spend Optimisation](#stage-7--marketing-spend-optimisation)
   - [Stage 8 — XAI Engine](#stage-8--xai-engine)
8. [Running the Pipeline](#running-the-pipeline)
9. [Module Reference](#module-reference)
10. [Outputs](#outputs)
11. [Configuration & Environment Variables](#configuration--environment-variables)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project solves the **DataStorm Round 2** challenge: given historical FMCG outlet transaction data and spatial signals, we must:

| Task | Description | Key Output |
|------|-------------|-----------|
| **Potential Prediction** | Estimate the latent maximum monthly sales volume for each outlet in January 2026 | `outputs/data_storm_predictions.csv` |
| **Budget Allocation** | Optimally allocate a LKR 5,000,000 promotional budget across Western Province outlets | `data_mavericks_budget_allocations.csv` |
| **Explainable AI** | Translate SFA model outputs into an executive-grade business narrative for any outlet | Console / JSON |

The analytical foundation is a **Stochastic Frontier Analysis (SFA)** model — an econometric technique that estimates the *latent demand ceiling* (production frontier) for each outlet and quantifies how far below that ceiling each outlet currently operates.

---

## Architecture

```
Raw CSV Files (data/raw/)
         │
         ▼
┌─────────────────────┐
│  Stage 1 · INGEST   │  Zero-transform copy to bronze layer
│  src/ingestion/     │
└─────────┬───────────┘
          │  data/bronze/*.csv
          ▼
┌─────────────────────┐
│  Stage 2 · CLEAN    │  Dedup, coord swap fix, referential integrity
│  src/cleaning/      │
└─────────┬───────────┘
          │  data/silver/dim_outlets.parquet
          │  data/silver/fact_transactions.parquet
          │  data/silver/dim_holidays.parquet
          │  data/silver/dim_distributor_seasonality.parquet
          ▼
┌─────────────────────┐
│  Stage 3 · POI      │  OSM spatial enrichment + competitor graph
│  src/poi/           │
└─────────┬───────────┘
          │  data/silver/silver_layer_poi_edges.parquet
          │  data/silver/silver_layer_competitor_graph.parquet
          ▼
┌─────────────────────┐
│  Stage 4 · FEATURES │  Merge + OHE + constraint proxies → Gold
│  src/features/      │
└─────────┬───────────┘
          │  data/gold/sfa_refined.parquet
          ▼
┌─────────────────────┐
│  Stage 5 · MODEL    │  MLE-fit SFA (half-normal inefficiency)
│  src/modeling/      │
└─────────┬───────────┘
          │  outputs/sfa_model.pkl
          │  outputs/outlet_potential_report_refined.parquet
          ▼
┌─────────────────────┐
│  Stage 6 · PREDICT  │  January 2026 latent potential per outlet
│  src/modeling/      │
└─────────┬───────────┘
          │  outputs/data_mavericks_predictions.csv
          ▼
┌─────────────────────┐
│  Stage 7 · OPTIMIZE │  Non-linear convex budget optimisation
│  src/optimization/  │
└─────────┬───────────┘
          │  data_mavericks_budget_allocations.csv
          │  outputs/data_mavericks_budget_allocations.csv
          │  reports/marketing_spend_summary.md
          ▼
┌─────────────────────┐
│  Stage 8 · XAI      │  Per-outlet LLM business narrative
│  src/x_ai/          │
└─────────────────────┘
          │  Console report / JSON
```

---

## Repository Structure

```
data-storm/
├── main.py                          # ← Pipeline orchestrator (start here)
├── pyproject.toml                   # Dependencies (managed by uv)
├── .env                             # API keys (GEMINI_API_KEY / GROQ_API_KEY)
│
├── data/
│   ├── raw/                         # Drop your source CSVs here
│   ├── bronze/                      # Exact copies of raw CSVs (auto-created)
│   ├── silver/                      # Cleaned & enriched Parquets (auto-created)
│   ├── gold/                        # SFA-ready model dataset (auto-created)
│   └── rejected/                    # Records that failed validation (auto-created)
│
├── outputs/                         # Model file + predictions (auto-created)
├── reports/                         # Generated Markdown reports (auto-created)
│
└── src/
    ├── ingestion/
    │   └── ingest_bronze.py         # Stage 1: raw → bronze
    ├── cleaning/
    │   ├── bronze_to_silver.py      # Stage 2: orchestrator
    │   └── cleaners.py              # Cleaning logic (transactions, outlets)
    ├── poi/
    │   ├── pipeline.py              # Stage 3: orchestrator
    │   ├── poi_decay.py             # Gaussian distance-decay POI edges
    │   └── competitor_graph.py      # Competitive catchment density graph
    ├── features/
    │   └── build_gold_sfa.py        # Stage 4: silver → gold
    ├── modeling/
    │   ├── sfa_model.py             # SFA model class (MLE, predict, save/load)
    │   ├── train_and_report_potential.py  # Stage 5: fit + report
    │   └── generate_predictions.py  # Stage 6: January 2026 predictions
    ├── optimization/
    │   └── market_spend_optim.py    # Stage 7: KKT dual bisection solver
    ├── validation/                  # Quality checks & anomaly detection
    │   ├── anomaly_detection.py
    │   ├── quality_checks.py
    │   ├── validate_silver.py
    │   └── verify_predictions.py
    └── x_ai/
        ├── schemas.py               # Pydantic models (payload & response)
        ├── engine.py                # SFA metric extraction & feature attribution
        ├── llm_service.py           # Gemini / Groq LLM call routing
        └── main.py                  # Stage 8: XAI orchestrator
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.13 | Managed via `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Fast package manager & virtual-env |
| OSM PBF file | — | `data/bronze/sri-lanka-260529.osm.pbf` (for POI stage) |
| Gemini or Groq API key | — | Required only for Stage 8 (XAI) |

> **Note on `pyrosm`**: The POI stage uses `pyrosm` to parse the `.osm.pbf` file. This library requires native compilation and may not install correctly on Windows. If it is unavailable and `data/silver/silver_layer_poi_edges.parquet` already exists (e.g. generated on Linux/macOS), the pipeline will automatically skip re-generation and use the existing file.

---

## Environment Setup

```powershell
# 1. Clone the repository
git clone <repo-url>
cd data-storm

# 2. Install uv (if not already installed)
pip install uv

# 3. Create virtual environment and install all dependencies
uv sync

# 4. Configure API keys
# Create/edit .env in the project root:
#   GEMINI_API_KEY=your_key_here      ← preferred
#   GROQ_API_KEY=your_key_here        ← fallback
```

**`.env` example:**
```dotenv
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

---

## Data Layout

Before running the pipeline, place your raw CSV source files inside `data/raw/`:

```
data/raw/
├── transactions_history_final.csv
├── outlet_master.csv
├── outlet_coordinates.csv
├── distributor_seasonality_details.csv
└── holiday_list.csv
```

Also place the OSM file in `data/bronze/` (needed for Stage 3):
```
data/bronze/
└── sri-lanka-260529.osm.pbf
```

> If you already have pre-generated Silver Parquets (e.g. from a previous run or a shared dataset), you can skip Stages 1-3 and jump directly to Stage 4.

---

## Full Pipeline Walkthrough

### Stage 1 — Bronze Ingestion

**Module:** [`src/ingestion/ingest_bronze.py`](src/ingestion/ingest_bronze.py)

Copies the five raw CSV files from `data/raw/` into `data/bronze/` with **zero transformations**. This creates an immutable audit trail of the original source data.

- Idempotent: files already present in bronze are skipped.
- Missing source files raise a warning (not an error) since bronze may have been pre-populated.

**Outputs:**
```
data/bronze/
├── transactions_history_final.csv
├── outlet_master.csv
├── outlet_coordinates.csv
├── distributor_seasonality_details.csv
└── holiday_list.csv
```

---

### Stage 2 — Silver Cleaning

**Module:** [`src/cleaning/bronze_to_silver.py`](src/cleaning/bronze_to_silver.py) + [`cleaners.py`](src/cleaning/cleaners.py)

Applies a series of data quality rules and writes cleaned Parquet files to `data/silver/`.

**Outlet Cleaning (`clean_outlets`):**
| Check | Action |
|-------|--------|
| Missing GPS coordinates | Reject to `rejected_outlets.csv` |
| Coordinates outside Sri Lanka bounds (lat 5.9–9.9, lon 79.6–81.9) | Attempt coordinate swap; reject if still invalid |
| Duplicate GPS location | Keep first occurrence; reject rest |
| Outlet type misspellings (e.g. `Grocry`) | Standardised via spelling map |
| Missing `Outlet_Size` | Filled as `"Unknown"` |

**Transaction Cleaning (`clean_transactions`):**
| Check | Action |
|-------|--------|
| Duplicate invoices (exact key match) | Reject to `rejected_transactions.csv` |
| Referential integrity (unknown `Outlet_ID`) | Reject to `rejected_transactions.csv` |

**Outputs:**
```
data/silver/
├── dim_outlets.parquet                  # Cleaned outlet master + coordinates
├── fact_transactions.parquet            # Deduplicated, validated transactions
├── dim_holidays.parquet                 # Pass-through holiday calendar
└── dim_distributor_seasonality.parquet  # Pass-through seasonality table

data/rejected/
├── rejected_outlets.csv
└── rejected_transactions.csv
```

---

### Stage 3 — POI & Competitor Graph

**Module:** [`src/poi/pipeline.py`](src/poi/pipeline.py)

This stage performs two parallel spatial enrichment tasks using outlet GPS coordinates:

#### 3a. Competitor Graph (`competitor_graph.py`)

Builds a bidirectional competitor catchment graph where every pair of outlets within **5 km** of each other is treated as competing. For each outlet, the following aggregate features are computed:

| Feature | Description |
|---------|-------------|
| `Competitor_Count_5km` | Number of competitor outlets within 5 km |
| `Min_Competitor_Distance_Meters` | Distance to the nearest competitor |
| `Total_Competitive_Friction` | Sum of friction scores from all nearby competitors |
| `Average_Competitive_Friction` | Mean friction score |

#### 3b. Spatial Distance-Decay POI Edges (`poi_decay.py`)

Parses the Sri Lanka `.osm.pbf` file and computes a **Gaussian distance-decay footfall impact score** for every (outlet, POI) pair within a 1 km micro-radius.

**Algorithm:**
1. Extract POIs matching a custom tag filter (schools, hospitals, bus stations, markets, etc.)
2. Build a `cKDTree` for fast spatial lookups
3. For each outlet:
   - **Macro query (3 km):** Count nearby POIs to compute a regional density multiplier (0.2× – 2.0×)
   - **Micro query (1 km):** Retrieve POIs for actual edge generation
4. Apply the decay formula:  
   `Footfall_Impact_Score = Base_Footfall × Regional_Multiplier × exp(−d² / 2σ²)`  
   where σ = 300 m

**Coordinate mode flag:**
```powershell
# Default: use uncorrected bronze coordinates (matches legacy Colab output)
uv run python main.py --step poi

# Use corrected silver-layer coordinates
uv run python main.py --step poi --corrected-coords
```

**Outputs:**
```
data/silver/
├── silver_layer_competitor_graph.parquet
└── silver_layer_poi_edges.parquet
```

> **Windows / pyrosm note:** If `pyrosm` is not installed and the output file already exists, this stage is gracefully skipped.

---

### Stage 4 — Gold Feature Engineering

**Module:** [`src/features/build_gold_sfa.py`](src/features/build_gold_sfa.py)

Merges all Silver datasets and engineers the final feature set for the SFA model.

**Pipeline steps:**

1. **Aggregate transactions** to monthly outlet-level totals (`Volume_Liters`, `Total_Bill_Value`)
2. **Reconstruct bidirectional competitor edges** and compute outlet-level graph aggregates
3. **Map POI types** to 7 business categories (Transport, Education, Health, Commercial, Residential, Social, Tourism) and sum footfall impact scores per category
4. **Merge** dimensions: outlets, competitors, POIs, seasonality, holidays
5. **Compute constraint proxies** per outlet:
   - `CV_Volume` — coefficient of variation in monthly volume (demand volatility)
   - `Flatline_Score` — proportion of months with the most frequent volume (supply rigidity)
   - `Round_Number_Bias` — ratio of volumes divisible by 50 or 100 (reporting bias)
   - `Price_Rigidity` — variance of price-per-liter over time
6. **Province extraction** from `Distributor_ID` prefix (W, C, S, NW, etc.)
7. **Log-transform** volume: `ln_volume = log(Volume_Liters)` (SFA requires log-space)
8. **One-hot encode** categorical features: `Outlet_Type`, `Outlet_Size`, `Province`, `Seasonality_Index` (with `drop_first=True` to avoid multicollinearity)
9. Filter to records with positive volume only

**Output:**
```
data/gold/
└── sfa_refined.parquet   # Model-ready dataset (~19,960 outlets × 36 months)
```

---

### Stage 5 — SFA Model Training

**Module:** [`src/modeling/train_and_report_potential.py`](src/modeling/train_and_report_potential.py)  
**Model class:** [`src/modeling/sfa_model.py`](src/modeling/sfa_model.py)

Fits a **Stochastic Frontier Analysis** model using Maximum Likelihood Estimation (MLE).

**Mathematical framework:**

```
ln(Volume_i) = X_i β + v_i − u_i

where:
  v_i ~ N(0, σ_v²)   — random noise (measurement error, weather)
  u_i ~ |N(0, σ_u²)| — non-negative inefficiency term (half-normal)
  
Frontier (maximum potential):
  Y*_i = exp(X_i β) × exp(0.5 σ_v²)   [log-normal bias correction]

Technical efficiency:
  TE_i = Y_actual_i / Y*_i   ∈ (0, 1]
```

**Key parameters reported:**
| Parameter | Meaning |
|-----------|---------|
| `sigma_u` | Systemic inefficiency std. dev. (supply constraints, credit limits) |
| `sigma_v` | Random noise std. dev. (weather, measurement) |
| `lambda = σ_u / σ_v` | Signal-to-noise ratio; higher = more systematic inefficiency |

**Features used as demand drivers:** All Gold layer features **except** constraint proxies (`CV_Volume`, `Flatline_Score`, `Round_Number_Bias`, `Price_Rigidity`) and metadata columns.

**Outputs:**
```
outputs/
├── sfa_model.pkl                              # Serialised SFAModel object
└── outlet_potential_report_refined.parquet    # Outlet-level efficiency report
```

---

### Stage 6 — Predictions

**Module:** [`src/modeling/generate_predictions.py`](src/modeling/generate_predictions.py)

Generates the **January 2026 latent potential** prediction for every outlet using a three-step uncapping logic grounded in SFA economic theory:

**Step 1 — Construct January 2026 feature vectors**  
Each outlet's static features are pulled from the Gold layer. The January distributor seasonality index (Moderate / Favorable) is applied, and holiday count is set to the historical January average (11.3 days).

**Step 2 — Group frontier prediction**  
```
Y*_group_jan2026 = exp(X_jan2026 × β) × exp(0.5 × σ_v²)
```

**Step 3 — Apply Systemic Inefficiency Multiplier**  
```
E[u] = σ_u × √(2/π)   [expected inefficiency for half-normal distribution]
Multiplier = exp(E[u])

Maximum_Monthly_Liters = max(Y*_group_jan2026,  Historical_Jan_Peak) × Multiplier
```

This guarantees predictions are **strictly above** each outlet's own historical January peak for 100% of outlets.

**Output:**
```
outputs/
└── data_mavericks_predictions.csv    # Outlet_ID | Maximum_Monthly_Liters
```

---

### Stage 7 — Marketing Spend Optimisation

**Module:** [`src/optimization/market_spend_optim.py`](src/optimization/market_spend_optim.py)

Formulates and solves a **non-linear convex optimisation** to allocate LKR 5,000,000 across Western Province outlets to maximise total volume lift.

**Objective function (logarithmic response curve — diminishing returns):**
```
Maximise:  Σ_i  a_i × ln(1 + b × x_i)

where:
  x_i  = spend allocated to outlet i  (decision variable)
  a_i  = historical normal January volume (= max(Y_historical_i, 1.0))
  b    = 0.0005  (spend scaling parameter)
```

**Constraints:**
```
1. Budget:      Σ_i x_i ≤ 5,000,000 LKR
2. SFA Ceiling: Lift_i = a_i × ln(1 + b × x_i) ≤ H_i   (headroom per outlet)
   → Translates to upper bound: x_i ≤ U_i = (exp(H_i / a_i) − 1) / b
3. Non-negativity: x_i ≥ 0
```

**Solver — KKT Dual Bisection:**  
The optimal solution has a closed-form structure (water-filling / Lagrangian relaxation):
```
x_i*(λ) = clip(a_i / λ − 1/b, 0, U_i)
```
The optimal Lagrange multiplier λ* is found via bisection on the budget constraint, making the solver **exact and extremely fast** (O(n log(1/ε))).

Fallback: SLSQP (scipy) if bisection fails.

**Outputs:**
```
data_mavericks_budget_allocations.csv          # Root (submission copy)
outputs/data_mavericks_budget_allocations.csv  # Outputs copy
reports/marketing_spend_summary.md             # Markdown breakdown by distributor/size/type
```

---

### Stage 8 — XAI Engine

**Module:** [`src/x_ai/`](src/x_ai/)

Generates a structured, human-readable **executive narrative** for any outlet by combining SFA model introspection with a large language model.

**Three-step orchestration (`src/x_ai/main.py`):**

```
Step 1 · get_engine_data()
  └─ Load sfa_model.pkl + retrieve outlet row from Gold Parquet

Step 2 · compute_xai_metrics()
  └─ Extract efficiency score, opportunity gap, feature impacts
  └─ Classify features into: environmental signals vs. operational constraints

Step 3 · generate_explanation()
  └─ Format structured JSON payload
  └─ Call Gemini 2.5 Flash (primary) or Groq LLaMA 3.3 70B (fallback)
  └─ Return 3-paragraph executive narrative
```

**Feature attribution methodology (`engine.py`):**
```
Percentage Impact_i = (exp(β_i) − 1) × 100
Local Driver Strength_i = Feature_Value_i × Percentage_Impact_i
```
Features are sorted by `Local_Driver_Strength` (descending) to surface the most impactful demand signals specific to the outlet.

**LLM narrative structure (system prompt enforces):**
| Paragraph | Content |
|-----------|---------|
| **The Score** | Latent ceiling vs. actual baseline; opportunity gap as investment thesis |
| **The Drivers** | Environmental & geospatial factors naturally elevating the outlet |
| **The Bottlenecks & Action** | Operational constraints + concrete spend recommendations |

**LLM routing (`llm_service.py`):**
- Primary: **Gemini 2.5 Flash** (if `GEMINI_API_KEY` is set)
- Fallback: **Groq LLaMA 3.3 70B Specdec** → LLaMA 3.1 8B Instant (if `GROQ_API_KEY` is set)

**Pydantic schemas (`schemas.py`):**
- `FeatureImpact` — per-feature attribution record
- `OutletXAIPayload` — full structured input sent to LLM
- `OutletXAIResponse` — final response with narrative + all metrics

---

## Running the Pipeline

### Quick Reference

```powershell
# Run a single stage
uv run python main.py --step ingest
uv run python main.py --step clean
uv run python main.py --step poi
uv run python main.py --step features
uv run python main.py --step model
uv run python main.py --step predict
uv run python main.py --step optimize         # default if no --step given
uv run python main.py --step xai --outlet-id OUT_00001

# Run full pipeline (stages 1-7) in one command
uv run python main.py --step all

# POI with corrected coordinates
uv run python main.py --step poi --corrected-coords

# XAI with JSON output
uv run python main.py --step xai --outlet-id OUT_00042 --output-json
```

### Recommended Execution Order (Fresh Run)

```powershell
# 1. Ingest raw data
uv run python main.py --step ingest

# 2. Clean and validate
uv run python main.py --step clean

# 3. Build spatial graph (requires pyrosm + OSM PBF file)
uv run python main.py --step poi

# 4. Engineer Gold features
uv run python main.py --step features

# 5. Train SFA model
uv run python main.py --step model

# 6. Generate predictions
uv run python main.py --step predict

# 7. Optimise marketing spend
uv run python main.py --step optimize

# 8. Explain a specific outlet (run any time after stage 5)
uv run python main.py --step xai --outlet-id OUT_00001
```

Or run everything at once:
```powershell
uv run python main.py --step all
```

---

## Module Reference

| Module | Entry Point | Key Functions |
|--------|-------------|---------------|
| `ingestion.ingest_bronze` | `ingest_bronze()` | Copies raw CSVs to bronze |
| `cleaning.bronze_to_silver` | `run_bronze_to_silver()` | Full bronze→silver pipeline |
| `cleaning.cleaners` | `clean_outlets()`, `clean_transactions()` | Per-dataset cleaning logic |
| `poi.pipeline` | `run_poi_and_competitor_pipeline()` | Orchestrates POI + graph |
| `poi.poi_decay` | `generate_poi_decay_edges()` | Gaussian decay + cKDTree |
| `poi.competitor_graph` | `generate_competitor_graph()` | 5 km catchment graph |
| `features.build_gold_sfa` | `build_refined_gold_layer()` | Silver → Gold feature engineering |
| `modeling.sfa_model` | `SFAModel` class | MLE fit, predict_potential, save/load |
| `modeling.train_and_report_potential` | `run_refined_sfa_pipeline()` | Train + report + save |
| `modeling.generate_predictions` | `generate_predictions()` | January 2026 predictions |
| `optimization.market_spend_optim` | `run_marketing_spend_optimization()` | KKT bisection solver |
| `x_ai.engine` | `compute_xai_metrics()` | SFA metric extraction |
| `x_ai.llm_service` | `generate_explanation()` | Gemini / Groq API call |
| `x_ai.main` | `generate_outlet_explanation()` | XAI end-to-end orchestrator |
| `x_ai.schemas` | Pydantic models | `FeatureImpact`, `OutletXAIPayload`, `OutletXAIResponse` |

---

## Outputs

| File | Stage | Description |
|------|-------|-------------|
| `data/bronze/*.csv` | 1 | Immutable raw data copies |
| `data/silver/dim_outlets.parquet` | 2 | Cleaned outlet master |
| `data/silver/fact_transactions.parquet` | 2 | Cleaned transaction history |
| `data/silver/dim_holidays.parquet` | 2 | Holiday calendar |
| `data/silver/dim_distributor_seasonality.parquet` | 2 | Seasonality indices |
| `data/silver/silver_layer_competitor_graph.parquet` | 3 | Competitor catchment graph |
| `data/silver/silver_layer_poi_edges.parquet` | 3 | POI decay edges |
| `data/gold/sfa_refined.parquet` | 4 | SFA-ready feature dataset |
| `outputs/sfa_model.pkl` | 5 | Trained SFA model (pickle) |
| `outputs/outlet_potential_report_refined.parquet` | 5 | Per-outlet efficiency scores |
| `outputs/data_mavericks_predictions.csv` | 6 | **Submission file** (Outlet_ID, Maximum_Monthly_Liters) |
| `data_mavericks_budget_allocations.csv` | 7 | **Submission file** (Outlet_ID, Trade_Spend_Allocation_LKR) |
| `reports/marketing_spend_summary.md` | 7 | Markdown spend breakdown report |
| Console / `--output-json` | 8 | XAI narrative + metrics |

---

## Configuration & Environment Variables

| Variable | Required | Used By | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Optional* | Stage 8 · XAI | Google Gemini 2.5 Flash API key |
| `GROQ_API_KEY` | Optional* | Stage 8 · XAI | Groq LLaMA API key (fallback) |

*At least one of `GEMINI_API_KEY` or `GROQ_API_KEY` **must** be set to run Stage 8.

Set these in your `.env` file at the project root:
```dotenv
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

**Pipeline constants (in-code):**

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `BUDGET_LKR` | `market_spend_optim.py` | 5,000,000 | Total promotional budget |
| `DEFAULT_B` | `market_spend_optim.py` | 0.0005 | Spend scaling parameter |
| `JAN_AVG_HOLIDAY_COUNT` | `generate_predictions.py` | 11.3 | Average Jan holidays (2023–2025) |
| `radius_micro` | `poi_decay.py` | 1000 m | POI edge radius |
| `radius_macro` | `poi_decay.py` | 3000 m | Density multiplier radius |
| `sigma` | `poi_decay.py` | 300 m | Gaussian decay sigma |

---

## Troubleshooting

### `FileNotFoundError: SFA model not found`
You need to run Stage 5 (model training) before Stage 6 or 8:
```powershell
uv run python main.py --step model
```

### `FileNotFoundError: data_storm_predictions.csv not found`
The optimization stage looks for the predictions CSV. Run Stage 6 first:
```powershell
uv run python main.py --step predict
```
> **Note:** The optimization module looks for `data_storm_predictions.csv` (without team prefix). If only `data_mavericks_predictions.csv` exists, rename it or copy it.

### `ImportError: pyrosm is not installed`
The POI stage requires `pyrosm` for parsing OSM files. On Windows this often fails to compile. **Workarounds:**
1. Use a pre-generated `silver_layer_poi_edges.parquet` (the stage will skip automatically if the file exists).
2. Run the POI stage on Linux/macOS or in WSL.
3. Install from a pre-built wheel if available for your Python version.

### `ValueError: No LLM API keys found`
Set at least one API key in `.env`:
```dotenv
GEMINI_API_KEY=AIza...
```

### `ValueError: Outlet ID OUT_XXXXX not found in the dataset`
The outlet ID provided to `--step xai` does not exist in `data/gold/sfa_refined.parquet`. Check valid IDs:
```python
import pandas as pd
df = pd.read_parquet("data/gold/sfa_refined.parquet")
print(df['Outlet_ID'].unique()[:20])
```

### Bisection fails during optimisation
If the KKT dual bisection fails, the optimizer automatically falls back to `scipy.optimize.minimize` with the SLSQP method. This is slower but produces correct results.

---

*Built with ❤️ by Team Data Mavericks for DataStorm Round 2.*
