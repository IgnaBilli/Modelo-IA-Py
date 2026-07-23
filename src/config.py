from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "terrabovina_dmi_kenya_raw.csv"
DATA_PATH = DATA_DIR / "processed" / "terrabovina_dmi_kenya_features V2.csv"
PROCESSED_DATA_PATH = DATA_PATH

MODEL_PATH = PROJECT_ROOT / "models" / "dmi_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_PATH = REPORTS_DIR / "model_metrics.json"
ETL_REPORT_PATH = REPORTS_DIR / "etl_summary.json"

