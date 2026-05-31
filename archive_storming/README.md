# DataStorm — Latent Outlet Potential Estimation

> **Competition:** DataStorm (Beverage Distributor Track)  
> **Objective:** Predict the Maximum Monthly Purchase Potential (litres) for 20,000 traditional trade outlets across 4 Sri Lankan provinces for **January 2026**.  
> **Approach:** Stochastic Frontier Analysis (SFA) on a Lakehouse pipeline (Bronze → Silver → Gold → Model → Predictions).

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Repository Structure](#2-repository-structure)  
3. [Prerequisites](#3-prerequisites)  
4. [Data Setup](#4-data-setup)  
5. [Running the Full Pipeline](#5-running-the-full-pipeline)  
6. [Running Individual Steps](#6-running-individual-steps)  
7. [Notebooks (EDA)](#7-notebooks-eda)  
8. [Output Files](#8-output-files)  
9. [Methodology Summary](#9-methodology-summary)  
10. [GenAI Transparency Log](#10-genai-transparency-log)

---

## 1. Project Overview

Traditional resource allocation (coolers, trade budgets, promotional discounts) in this distribution network is based on **historical average sales**, which only reflects what an outlet *did* sell — not what it *could* sell.

This pipeline estimates the **latent (uncapped) demand ceiling** for each outlet by:

1. **Data Forensics** — Detecting and quarantining system artefacts (ghost entries, delivery caps, swapped GPS, duplicates) in the raw SFA/ERP exports.
2. **Feature Engineering** — Building demand drivers (outlet type, POI catchment, holiday density, distributor seasonality) and constraint proxies (flatline score, round-number bias, volume CV).
3. **Stochastic Frontier Analysis (SFA)** — Fitting a production frontier in log-space. The frontier represents the *maximum achievable volume* given an outlet's demand characteristics. The gap between the frontier and observed volume quantifies systemic constraints (credit limits, stockouts, delivery caps).
4. **January 2026 Projection** — Adjusting the base potential with distributor-level seasonality indices.

---

## 2. Repository Structure

```
data-storm/
│
├── main.py                          # ← Pipeline orchestrator (run this)
├── pyproject.toml                   # Python project config (uv)
├── README.md
│
├── data/
│   ├── raw/                         # (Optional) Place original CSVs here before ingestion
│   ├── bronze/                      # Raw files copied AS-IS (no transformation)
│   ├── silver/                      # Cleaned, validated datasets (parquet)
│   ├── gold/                        # Feature-engineered, SFA-ready datasets (parquet)
│   └── rejected/                    # Quarantined records with documented failure reasons (CSV)
│
├── src/
│   ├── ingestion/
│   │   └── ingest_bronze.py         # STEP 1: Copy raw CSVs → data/bronze
│   │
│   ├── cleaning/
│   │   ├── bronze_to_silver.py      # STEP 2: Orchestrates cleaning for all datasets
│   │   └── cleaners.py              # Cleaning logic (transactions, outlets)
│   │
│   ├── validation/
│   │   ├── quality_checks.py        # Reusable DQ functions (nulls, dupes, ranges, RI)
│   │   ├── anomaly_detection.py     # Domain-specific anomaly detectors
│   │   ├── validate_silver.py       # STEP 3a: Silver layer QC report
│   │   └── run_anomaly_detection.py # STEP 3b: Anomaly report on raw bronze data
│   │
│   ├── features/
│   │   └── build_gold_sfa.py        # STEP 4: Feature engineering → sfa_refined.parquet
│   │
│   ├── modeling/
│   │   ├── sfa_model.py             # SFA model class (log-likelihood, fit, predict)
│   │   ├── train_and_report_potential.py  # STEP 5: Train SFA, save outlet report
│   │   └── generate_predictions.py  # STEP 6: Produce final competition CSV
│   │
│   └── poi/                         # (Placeholder) POI scraping scripts go here
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb   # EDA on bronze datasets
│   ├── data_validation.ipynb             # Interactive data quality exploration
│   ├── silver_layer_eda.ipynb            # Post-cleaning analysis
│   └── geographic_and_noise_analysis.ipynb  # Geospatial & noise EDA
│
├── outputs/
│   ├── sfa_model.pkl                     # Trained SFA model (pickle)
│   ├── outlet_potential_report_refined.parquet  # Full outlet-level SFA results
│   └── data_storm_predictions.csv        # ← FINAL SUBMISSION FILE
│
└── reports/                         # (For PDF summary report assets)
```

---

## 3. Prerequisites

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management.

### Install uv (if not installed)

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### Create environment and install dependencies

```powershell
cd C:\data-storm
uv sync
```

This installs all dependencies listed in `pyproject.toml` into a local `.venv`.

**Key dependencies:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `geopandas`, `folium`, `matplotlib`, `seaborn`

---

## 4. Data Setup

The raw data files must be present in **`data/bronze/`** before running the pipeline.

### Option A — Files already in `data/bronze/` (most likely)

If the 5 CSV files are already in place, skip to [Step 5](#5-running-the-full-pipeline). The pipeline will detect them automatically.

```
data/bronze/
├── transactions_history_final.csv
├── outlet_master.csv
├── outlet_coordinates.csv
├── distributor_seasonality_details.csv
└── holiday_list.csv
```

### Option B — Copy from a different source directory

Place the raw files in `data/raw/` and run:

```powershell
uv run python main.py --step bronze
```

Or specify a custom source path:

```powershell
uv run python src/ingestion/ingest_bronze.py "C:\path\to\raw\files"
```

> **Important:** The Bronze layer **never modifies** the source files. It is an append-only, read-only snapshot of the original data.

### POI Data

After you upload the POI scraping script to `src/poi/`, place the resulting enriched file at:

```
data/silver/outlets_with_poi_counts.csv
```

Columns required: `Outlet_ID`, `POI_Count_500m`, `POI_Count_1km`

---

## 5. Running the Full Pipeline

> **One command to reproduce all results end-to-end:**

```powershell
cd C:\data-storm
uv run python main.py
```

This sequentially runs all 6 steps:

| Step | Name | Input | Output |
|------|------|-------|--------|
| 1 | **Bronze** | `data/raw/*.csv` | `data/bronze/*.csv` (no-op if already there) |
| 2 | **Silver** | `data/bronze/*.csv` | `data/silver/*.parquet`, `data/rejected/*.csv` |
| 3 | **Validate** | `data/silver/*.parquet` | Console QC report + anomaly summary |
| 4 | **Gold** | `data/silver/*.parquet` | `data/gold/sfa_refined.parquet` |
| 5 | **Model** | `data/gold/sfa_refined.parquet` | `outputs/sfa_model.pkl`, `outputs/outlet_potential_report_refined.parquet` |
| 6 | **Predict** | `outputs/outlet_potential_report_refined.parquet` | `outputs/data_storm_predictions.csv` |

**Expected runtime:** ~5–15 minutes on a modern laptop (the SFA optimisation dominates).

---

## 6. Running Individual Steps

Use `--step` to run any single stage:

```powershell
# Re-clean data after adjusting cleaning logic
uv run python main.py --step silver

# Re-run only feature engineering
uv run python main.py --step gold

# Retrain model only
uv run python main.py --step model

# Regenerate submission CSV without retraining
uv run python main.py --step predict
```

Or run individual scripts directly:

```powershell
# Bronze ingestion
uv run python src/ingestion/ingest_bronze.py

# Silver cleaning
uv run python src/cleaning/bronze_to_silver.py

# Silver validation report
uv run python src/validation/validate_silver.py

# Anomaly detection report
uv run python src/validation/run_anomaly_detection.py

# Gold feature engineering
uv run python src/features/build_gold_sfa.py

# SFA model training
uv run python src/modeling/train_and_report_potential.py

# Final predictions
uv run python src/modeling/generate_predictions.py
```

---

## 7. Notebooks (EDA)

Notebooks are for **exploration and interpretation only** — they do not need to be run to reproduce the final predictions. Open them with Jupyter:

```powershell
uv run jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `exploratory_data_analysis.ipynb` | Initial profiling of all bronze datasets |
| `data_validation.ipynb` | Interactive DQ check exploration |
| `silver_layer_eda.ipynb` | Post-cleaning data distributions |
| `geographic_and_noise_analysis.ipynb` | Geospatial anomaly mapping + noise characterisation |

---

## 8. Output Files

| File | Description |
|------|-------------|
| `data/rejected/rejected_outlets.csv` | Outlets quarantined with `failure_reason` column |
| `data/rejected/rejected_transactions.csv` | Transactions quarantined with `failure_reason` column |
| `data/silver/fact_transactions.parquet` | Cleaned transaction fact table |
| `data/silver/dim_outlets.parquet` | Cleaned outlet dimension (master + coordinates merged) |
| `data/gold/sfa_refined.parquet` | Feature-engineered, model-ready dataset |
| `outputs/sfa_model.pkl` | Serialised trained SFA model |
| `outputs/outlet_potential_report_refined.parquet` | Outlet-level potential + efficiency scores |
| **`outputs/data_storm_predictions.csv`** | **← Final submission: `Outlet_ID`, `Maximum_Monthly_Liters`** |

---

## 9. Methodology Summary

### 9.1 Data Forensics & Quality Checks

Reusable, parameterisable functions in `src/validation/quality_checks.py`:

| Check | Function | Applied To |
|-------|----------|------------|
| **Null Check** | `check_nulls(df, columns)` | All mandatory fields |
| **Duplicate Check** | `check_duplicates(df, keys)` | Composite transaction keys; Outlet_ID |
| **Range Check** | `check_ranges(df, col, min, max)` | Volume_Liters ≥ 0; GPS bounds |
| **Referential Integrity** | `check_referential_integrity(child, parent, key)` | Transaction → Outlet master |
| **Type/Format Check** | `check_datatypes(df, schema)` | Date fields, ID formats |

Domain-specific anomaly detectors in `src/validation/anomaly_detection.py`:

- Swapped GPS coordinates (corrected automatically when valid after swap)
- Outlets outside Sri Lanka bounding box (5.9°–9.9°N, 79.6°–81.9°E)
- Duplicate exact GPS points (potential ghost/copied entries)
- Delivery cap patterns (same total volume repeated across months)
- Midnight batch upload timestamps
- Extreme sales volatility (CV > 2.0)

### 9.2 Feature Engineering (Gold Layer)

| Feature Category | Features |
|-----------------|----------|
| **Demand Drivers** | Outlet_Type (OHE), Outlet_Size (OHE), Province (OHE), POI_Count_500m, POI_Count_1km, Holiday_Count |
| **Seasonality** | Distributor-level Seasonality_Index (OHE bucketed) |
| **Constraint Proxies** (not in frontier) | CV_Volume, Flatline_Score, Round_Number_Bias, Price_Rigidity |
| **Target** | `ln_volume` = log(Volume_Liters) |

### 9.3 Stochastic Frontier Analysis (SFA)

The SFA model is defined in `src/modeling/sfa_model.py`. It fits:

```
ln(Y_observed) = X·β + v - u

  where:
    X·β  = production frontier (log-space demand ceiling)
    v    ~ N(0, σ_v²)   — random noise (measurement error, weather, etc.)
    u    ~ |N(0, σ_u²)| — one-sided inefficiency term (credit limits, stockouts, caps)
```

**Maximum Likelihood Estimation** is performed via `scipy.optimize.minimize` with L-BFGS-B.

The **potential (frontier) volume** for outlet *i* is:

```
Y_potential = exp(X_i · β̂) × exp(0.5 · σ̂_v²)
```

The **efficiency score** measures how close an outlet operates to its ceiling:

```
TE_i = Y_observed_i / Y_potential_i  ∈ (0, 1]
```

**January 2026 adjustment:**  
`Maximum_Monthly_Liters = Y_potential × Seasonality_Index(Jan)`

---

## 10. GenAI Transparency Log

| Tool | Where Used | How Used |
|------|-----------|----------|
| **Antigravity (Google DeepMind)** | Pipeline architecture | Accelerated scaffolding of the Bronze/Silver/Gold Lakehouse structure, reusable DQ check functions, and the SFA log-likelihood implementation. Code was reviewed and validated manually. |
| **Antigravity** | Anomaly detection | Suggested domain-specific checks (midnight batch uploads, delivery cap detection, round-number bias) based on FMCG SFA domain knowledge. |
| **Antigravity** | Feature engineering | Co-designed constraint proxy features (flatline score, price rigidity) and the log-space transformation strategy for SFA. |
| **Antigravity** | Debugging | Used to diagnose numerical instability in the SFA optimizer and resolve SciPy convergence warnings. |
| **Antigravity** | README & documentation | Generated structured documentation templates; all content was verified against the actual implementation. |

> **All AI-generated code was validated against the competition guidelines and manually reviewed for correctness before inclusion.**

---

*DataStorm Team | May 2026*
