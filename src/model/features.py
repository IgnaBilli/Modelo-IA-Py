import pandas as pd

TARGET = "dmi_kg_dia"
GROUP = "hh"
FEATURES = [
    "peso_vivo_kg",
    "categoria_terrabovina",
    "condicion_corporal",
    "estado_fisiologico_norm",
]
NUMERIC_FEATURES = ["peso_vivo_kg", "condicion_corporal"]
CATEGORICAL_FEATURES = ["categoria_terrabovina", "estado_fisiologico_norm"]
CATEGORIES = ["Vaca", "Vaquilla", "Novillo", "Toro", "Ternero/Ternera"]
MISSING_CATEGORY = "DESCONOCIDO"


def normalize_estado(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING_CATEGORY
    normalized = str(value).strip().upper().replace(" ", "")
    return normalized or MISSING_CATEGORY


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same business transformations for training and inference."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    out = df[FEATURES].copy()
    out["peso_vivo_kg"] = pd.to_numeric(out["peso_vivo_kg"], errors="raise")
    out["condicion_corporal"] = pd.to_numeric(out["condicion_corporal"], errors="coerce")
    out["categoria_terrabovina"] = out["categoria_terrabovina"].astype(str).str.strip()
    out["estado_fisiologico_norm"] = out["estado_fisiologico_norm"].map(normalize_estado)

    not_cow = out["categoria_terrabovina"] != "Vaca"
    out.loc[not_cow, "condicion_corporal"] = float("nan")
    out.loc[not_cow, "estado_fisiologico_norm"] = MISSING_CATEGORY
    return out


def load_training_data(path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET, GROUP])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset invalido; faltan columnas: {sorted(missing)}")
    if len(df) < 1000:
        raise ValueError(f"Dataset insuficientemente poblado: se encontraron {len(df)} filas.")


    X = prepare_features(df)
    y = pd.to_numeric(df[TARGET], errors="raise")
    groups = df[GROUP]
    if not y.between(0.2, 20).all() or not X["peso_vivo_kg"].between(15, 900).all():
        raise ValueError("El dataset contiene valores fuera de los rangos validados por el ETL")
    return X, y, groups
