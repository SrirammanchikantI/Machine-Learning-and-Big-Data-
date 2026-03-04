# Kickstarter Campaign Success Prediction
### Big Data & Machine Learning Assignment

**Problem:** Binary classification — predict whether a Kickstarter campaign will succeed (`state=1`) or fail (`state=0`).  
**Dataset:** `kickstarter_2022-2021_unique_blurbs.csv` (~150k rows, 20+ features)  
**Stack:** PySpark 3.5, MLlib, scikit-learn (baseline), Python 3.10

---

## Project Structure

```
├── notebooks/
│   ├── 1_data_ingestion.ipynb       # Spark ingestion, validation, Parquet write
│   ├── 2_feature_engineering.ipynb  # Feature transforms, custom transformer
│   ├── 3_model_training.ipynb       # LR, RF, GBT + CrossValidator
│   └── 4_evaluation.ipynb           # Metrics, sklearn baseline, scalability plots
├── scripts/
│   ├── data_engineering.py          # Standalone ingestion script
│   ├── run_pipeline.py              # End-to-end pipeline orchestration
│   └── performance_profiler.py      # Strong / weak scaling experiments
├── config/
│   └── spark_config.yaml            # Spark tuning config with justifications
├── data/
│   ├── kickstarter_2022-2021_unique_blurbs.csv
│   └── parquet/kickstarter/         # Partitioned by country (generated)
├── models/
│   └── best_gbt_model/              # Saved CrossValidator GBT model
├── output/
│   └── scaling_report.json          # Performance profiler results
├── tests/
│   └── test_pipeline.py             # Unit + integration tests
├── main.py                          # Full PySpark MLlib training pipeline
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup

### Option A — pip (venv)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate ml_bigdata
```

> Java 11+ must be installed and `JAVA_HOME` set for PySpark to work.

---

## Running the Pipeline

### Step 1 — Data Engineering (CSV → Parquet)
```bash
python scripts/data_engineering.py
```

### Step 2 — ML Training (LR + RF + GBT + CrossValidator)
```bash
python main.py
```

### Full pipeline in one command
```bash
python scripts/run_pipeline.py
```

### Run tests
```bash
python tests/test_pipeline.py
```

### Scalability profiler
```bash
python scripts/performance_profiler.py
# Results saved to output/scaling_report.json
```

---

## ML Algorithms

| Algorithm | Library | AUC (approx) |
|-----------|---------|-------------|
| Logistic Regression | PySpark MLlib | ~0.73 |
| Random Forest | PySpark MLlib | ~0.78 |
| Gradient Boosted Trees (CV) | PySpark MLlib | ~0.80 |
| Logistic Regression (baseline) | scikit-learn | ~0.71 |

---

## Technical Highlights

- **Storage:** Parquet (Snappy) partitioned by `country` — enables predicate pushdown for country-filtered queries
- **Custom Transformer:** `LogGoalTransformer` applies `log1p` to `goal` column inside the ML Pipeline
- **Hyperparameter Tuning:** `CrossValidator` with 3-fold CV, `parallelism=2`
- **Caching:** `persist(MEMORY_AND_DISK)` on preprocessed DataFrame, `unpersist()` after write
- **Model Serialization:** GBT saved via `MLlib` (`.save()`); sklearn model via `pickle`
- **Scalability:** Strong & weak scaling experiments in `performance_profiler.py`
