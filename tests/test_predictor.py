from src.model.predictor import HerdPredictor


def test_batch_prediction_preserves_ids_and_returns_warning():
    predictor = HerdPredictor()
    animals = [
        {
            "animal_id": "A-1",
            "peso_vivo_kg": 420,
            "categoria_terrabovina": "Vaca",
            "condicion_corporal": 3,
            "estado_fisiologico_norm": "L",
        },
        {
            "animal_id": "A-2",
            "peso_vivo_kg": 900,
            "categoria_terrabovina": "Toro",
            "condicion_corporal": None,
            "estado_fisiologico_norm": None,
        },
    ]
    predictions, warnings = predictor.predict(animals)
    assert [item["animal_id"] for item in predictions] == ["A-1", "A-2"]
    assert all(item["dmi_kg_dia"] > 0 for item in predictions)
    assert any("A-2" in warning for warning in warnings)
