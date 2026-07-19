import pandas as pd

from src.model.features import MISSING_CATEGORY, prepare_features


def test_non_cow_optional_features_are_ignored():
    data = pd.DataFrame([{
        "peso_vivo_kg": 250,
        "categoria_terrabovina": "Novillo",
        "condicion_corporal": 3.5,
        "estado_fisiologico_norm": " l ",
    }])
    result = prepare_features(data)
    assert pd.isna(result.loc[0, "condicion_corporal"])
    assert result.loc[0, "estado_fisiologico_norm"] == MISSING_CATEGORY


def test_cow_physiological_state_is_normalized():
    data = pd.DataFrame([{
        "peso_vivo_kg": 400,
        "categoria_terrabovina": "Vaca",
        "condicion_corporal": 3,
        "estado_fisiologico_norm": " p/ l ",
    }])
    assert prepare_features(data).loc[0, "estado_fisiologico_norm"] == "P/L"
