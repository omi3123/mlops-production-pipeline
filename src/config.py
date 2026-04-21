from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "reference_sample.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production" / "production_events.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
MODEL_META_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "top_features.csv"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.csv"
DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift_summary.json"
BATCH_REPORT_PATH = PROJECT_ROOT / "reports" / "batch_monitoring.csv"
SCORED_HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "scored_holdout.csv"
