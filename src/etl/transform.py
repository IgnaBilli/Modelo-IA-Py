import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CATEGORIES = ["Vaca", "Vaquilla", "Novillo", "Toro", "Ternero/Ternera"]
MISSING_CATEGORY = "DESCONOCIDO"


def normalize_estado(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING_CATEGORY
    normalized = str(value).strip().upper().replace(" ", "")
    return normalized or MISSING_CATEGORY


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Perform thorough data cleaning, outlier handling, categorical normalization,
    and business rule enforcement.
    Returns (cleaned_df, transform_stats).
    """
    logger.info("Iniciando transformación y limpieza de datos...")
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Outlier removal if flagged in dataset
    outlier_cols = ["outlier_peso", "outlier_dmi_abs", "outlier_dmi_pct_bw"]
    outliers_removed = 0
    existing_outlier_cols = [c for c in outlier_cols if c in df_clean.columns]
    if existing_outlier_cols:
        outlier_mask = df_clean[existing_outlier_cols].any(axis=1)
        outliers_removed = int(outlier_mask.sum())
        df_clean = df_clean[~outlier_mask].copy()
        logger.info(f"Filtro de outliers pre-etiquetados: {outliers_removed} filas removidas.")

    # 2. Type conversions and numeric parsing
    df_clean["peso_vivo_kg"] = pd.to_numeric(df_clean["peso_vivo_kg"], errors="coerce")
    df_clean["dmi_kg_dia"] = pd.to_numeric(df_clean["dmi_kg_dia"], errors="coerce")
    df_clean["condicion_corporal"] = pd.to_numeric(df_clean["condicion_corporal"], errors="coerce")

    # 3. Categorical normalization
    df_clean["categoria_terrabovina"] = df_clean["categoria_terrabovina"].astype(str).str.strip()
    df_clean["estado_fisiologico_norm"] = df_clean["estado_fisiologico_norm"].map(normalize_estado)

    # 4. Domain range filtering
    valid_range_mask = (
        df_clean["peso_vivo_kg"].between(15.0, 900.0) &
        df_clean["dmi_kg_dia"].between(0.2, 20.0) &
        df_clean["categoria_terrabovina"].isin(CATEGORIES)
    )
    invalid_domain_count = int((~valid_range_mask).sum())
    df_clean = df_clean[valid_range_mask].copy()

    # 5. Nutritional business rules: Non-cow categories don't have body condition or lactation state
    not_cow = df_clean["categoria_terrabovina"] != "Vaca"
    df_clean.loc[not_cow, "condicion_corporal"] = np.nan
    df_clean.loc[not_cow, "estado_fisiologico_norm"] = MISSING_CATEGORY

    # 6. Feature Engineering: % DMI over body weight
    df_clean["dmi_pct_peso_vivo"] = (df_clean["dmi_kg_dia"] / df_clean["peso_vivo_kg"]) * 100.0

    final_rows = len(df_clean)
    stats = {
        "initial_rows": initial_rows,
        "outliers_removed": outliers_removed,
        "invalid_domain_rows_removed": invalid_domain_count,
        "final_rows": final_rows,
        "rows_retained_pct": round((final_rows / initial_rows) * 100.0, 2),
        "categories_distribution": df_clean["categoria_terrabovina"].value_counts().to_dict(),
        "weight_stats_by_category": {
            cat: {
                "min": float(df_clean.loc[df_clean["categoria_terrabovina"] == cat, "peso_vivo_kg"].min()),
                "mean": round(float(df_clean.loc[df_clean["categoria_terrabovina"] == cat, "peso_vivo_kg"].mean()), 2),
                "max": float(df_clean.loc[df_clean["categoria_terrabovina"] == cat, "peso_vivo_kg"].max()),
            }
            for cat in CATEGORIES if cat in df_clean["categoria_terrabovina"].values
        },
    }

    logger.info(f"Transformación completada: {final_rows} filas válidas conservadas ({stats['rows_retained_pct']}%).")
    return df_clean, stats
