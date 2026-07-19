from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "terrabovina_dmi_kenya_features V2.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "dmi_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
