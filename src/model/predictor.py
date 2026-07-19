import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.model.features import prepare_features


class HerdPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecute: python -m src.model.train")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.metadata = bundle["metadata"]

    def predict(self, animals: list[dict]) -> tuple[list[dict], list[str]]:
        frame = pd.DataFrame(animals)
        X = prepare_features(frame)
        dmi_values = np.maximum(self.model.predict(X), 0.0)
        predictions = []
        warnings = set()

        for animal, dmi in zip(animals, dmi_values):
            category = animal["categoria_terrabovina"]
            weight = float(animal["peso_vivo_kg"])
            domain = self.metadata["weight_ranges_kg"][category]
            if not domain["min"] <= weight <= domain["max"]:
                warnings.add(
                    f"{animal['animal_id']}: peso {weight:g} kg fuera del rango observado para {category} "
                    f"({domain['min']:g}-{domain['max']:g} kg)."
                )
            predictions.append({
                "animal_id": animal["animal_id"],
                "dmi_kg_dia": round(float(dmi), 2),
                "dmi_pct_peso_vivo": round(float(dmi / weight * 100), 2),
            })
        return predictions, sorted(warnings)
