import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    df: pd.DataFrame,
    output_csv_path: Path,
    etl_report_path: Path,
    metadata: dict,
) -> None:
    """
    Save cleaned data to target CSV and dump full ETL summary JSON report.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    etl_report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Guardando dataset procesado en: {output_csv_path}")
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    logger.info(f"Guardando reporte de calidad ETL en: {etl_report_path}")
    etl_report_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Carga y guardado de artefactos completados exitosamente.")
