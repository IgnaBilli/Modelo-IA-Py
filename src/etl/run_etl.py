from datetime import datetime, timezone
import logging

from src.config import DATA_PATH, ETL_REPORT_PATH, RAW_DATA_PATH
from src.etl.extract import extract_data
from src.etl.load import load_data
from src.etl.transform import transform_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_etl_pipeline() -> dict:
    """
    Execute end-to-end Extract, Transform, Load pipeline.
    """
    logger.info("=== INICIANDO PIPELINE ETL TERRABOVINA DMI ===")
    source = RAW_DATA_PATH if RAW_DATA_PATH.exists() else DATA_PATH

    raw_df, extract_stats = extract_data(source)
    clean_df, transform_stats = transform_data(raw_df)

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "extract": extract_stats,
        "transform": transform_stats,
    }

    load_data(clean_df, DATA_PATH, ETL_REPORT_PATH, metadata)
    logger.info("=== PIPELINE ETL FINALIZADO EXITOSAMENTE ===")
    return metadata


if __name__ == "__main__":
    run_etl_pipeline()
