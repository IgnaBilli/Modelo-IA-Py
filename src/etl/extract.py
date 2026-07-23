import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "animal_id",
    "hh",
    "categoria_terrabovina",
    "peso_vivo_kg",
    "dmi_kg_dia",
    "condicion_corporal",
    "estado_fisiologico_norm",
]


def extract_data(source_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Extract raw data from source CSV path and validate schema contract.
    Returns (DataFrame, metadata_dict).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Archivo de origen no encontrado en {source_path}")

    logger.info(f"Extrayendo datos de: {source_path}")
    df = pd.read_csv(source_path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el archivo fuente: {sorted(missing_cols)}")

    stats = {
        "source_path": str(source_path),
        "total_rows_extracted": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "null_counts_raw": df[REQUIRED_COLUMNS].isnull().sum().to_dict(),
    }

    logger.info(f"Extracción completada. {len(df)} filas cargadas de {len(df.columns)} columnas.")
    return df, stats
