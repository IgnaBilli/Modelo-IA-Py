import numpy as np
import pandas as pd

from src.model.features import CATEGORIES


LACTATING_STATES = {"L", "P/L", "B/L"}
PREGNANT_STATES = {"P", "P/L", "N/P"}
DRY_STATES = {"B", "B/L"}


def add_physiological_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create shared physiological signals and category-specific weight curves."""
    out = frame.copy()
    weight = out["peso_vivo_kg"].astype(float)
    condition = out["condicion_corporal"].astype(float)
    state = out["estado_fisiologico_norm"].astype(str)

    out["peso_vivo_kg_sq"] = weight**2
    out["peso_vivo_kg_sqrt"] = np.sqrt(weight.clip(lower=0))
    out["peso_x_condicion"] = weight * condition
    out["esta_lactando"] = state.isin(LACTATING_STATES).astype(float)
    out["esta_preniada"] = state.isin(PREGNANT_STATES).astype(float)
    out["esta_seca"] = state.isin(DRY_STATES).astype(float)
    out["estado_normal"] = (state == "N").astype(float)
    out["estado_desconocido"] = (state == "DESCONOCIDO").astype(float)
    out["peso_x_lactando"] = weight * out["esta_lactando"]
    out["peso_x_preniada"] = weight * out["esta_preniada"]

    for category in CATEGORIES:
        out[f"peso_categoria_{category}"] = weight.where(
            out["categoria_terrabovina"] == category,
            0.0,
        )
    return out


class CategoryBlendRegressor:
    """Route each category through a validated blend of two fitted regressors."""

    def __init__(self, models: dict, routing: dict, baseline_name: str):
        self.models = models
        self.routing = routing
        self.baseline_name = baseline_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        baseline = np.asarray(self.models[self.baseline_name].predict(X), dtype=float)
        predictions = baseline.copy()
        categories = X["categoria_terrabovina"].to_numpy()
        specialist_cache = {}

        for category, route in self.routing.items():
            specialist_name = route["specialist"]
            weight = float(route["specialist_weight"])
            if weight <= 0 or specialist_name == self.baseline_name:
                continue
            if specialist_name not in specialist_cache:
                specialist_cache[specialist_name] = np.asarray(
                    self.models[specialist_name].predict(X),
                    dtype=float,
                )
            mask = categories == category
            predictions[mask] = (
                (1.0 - weight) * baseline[mask]
                + weight * specialist_cache[specialist_name][mask]
            )
        return np.maximum(predictions, 0.0)
