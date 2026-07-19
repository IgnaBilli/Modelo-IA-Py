# Modelo DMI TerraBovina

Modelo Python para estimar los kilogramos de materia seca que consume un bovino por dia.

## Estructura

```text
data/processed/terrabovina_dmi_kenya_features V2.csv  Dataset de entrenamiento
src/model/features.py                                  Preparacion de variables
src/model/train.py                                     Evaluacion y entrenamiento
src/model/predictor.py                                 Prediccion local por lote
models/dmi_model.joblib                                Modelo entrenado
reports/model_metrics.json                             Metricas y metadatos
tests/                                                 Pruebas automaticas
```

No hay API, Docker ni datasets intermedios. El CSV V2 es la unica fuente de datos.

## Dataset

El archivo contiene 3.840 mediciones de bovinos de Bomet, Kenia. Sus decimales fueron reparados usando la salida canonica del ETL y se verifico que:

- El esquema y las 3.840 observaciones coincidieran fila por fila.
- El peso estuviera entre 15 y 900 kg.
- El DMI estuviera entre 0,2 y 20 kg/dia.
- `dmi_pct_peso_vivo = dmi_kg_dia / peso_vivo_kg * 100`.

Solo se conserva el V2 reparado; las copias usadas para verificarlo fueron eliminadas.

## Variables

Objetivo:

```text
dmi_kg_dia
```

Entradas del modelo:

```text
peso_vivo_kg
categoria_terrabovina
condicion_corporal
estado_fisiologico_norm
```

La condicion corporal y el estado fisiologico solo se usan para vacas, porque en las demas categorias tienen demasiados faltantes. Para otros animales se fuerzan a valores ausentes. El estado fisiologico se normaliza eliminando espacios y convirtiendolo a mayusculas.

No se usan identificadores, fechas, periodo ni `dmi_pct_peso_vivo`. Esta ultima columna se calcula con el objetivo y produciria fuga de informacion.

## Entrenamiento

Se comparan tres alternativas:

- `DummyRegressor`: promedio global, usado como referencia minima.
- `Ridge`: relacion lineal regularizada.
- `CatBoostRegressor`: modelo de arboles que maneja categorias directamente.

La evaluacion usa cinco folds de `StratifiedGroupKFold`. Todas las filas de un hogar (`hh`) quedan en un solo lado del split y se conserva aproximadamente la proporcion de categorias. El ganador es el modelo con menor MAE fuera de muestra y luego se entrena con las 3.840 filas.

```bash
python -m pip install -r requirements.txt
python -m src.model.train
```

El comando actualiza `models/dmi_model.joblib` y `reports/model_metrics.json`.
Durante la ejecucion se muestra una barra con el porcentaje completado, el modelo y el fold que se estan procesando.

## Prediccion

```python
from src.model.predictor import HerdPredictor

predictor = HerdPredictor()
predicciones, advertencias = predictor.predict([
    {
        "animal_id": "A-001",
        "peso_vivo_kg": 420,
        "categoria_terrabovina": "Vaca",
        "condicion_corporal": 3.0,
        "estado_fisiologico_norm": "L",
    },
    {
        "animal_id": "A-002",
        "peso_vivo_kg": 280,
        "categoria_terrabovina": "Novillo",
        "condicion_corporal": None,
        "estado_fisiologico_norm": None,
    },
])
```

Cada resultado incluye `animal_id`, `dmi_kg_dia` y `dmi_pct_peso_vivo`. También se informa si el peso queda fuera del rango observado para su categoria.

Para probar el modelo desde un menu interactivo:

```bash
python pruebaModelo.py
```

El programa permite cargar uno o varios animales y muestra el consumo estimado junto con las advertencias correspondientes.

Tambien se puede probar mediante archivos JSON. Edite `testReal/input.json` y ejecute:

```bash
python testReal/ejecutar.py
```

El resultado se guarda en `testReal/output.json`.

## Pruebas

```bash
python -m pytest
```

## Limitacion principal

Los datos proceden de Kenia y no incluyen clima, calidad ni disponibilidad del forraje. Las metricas miden generalizacion entre hogares de ese dataset, no garantizan precision en sistemas argentinos. Se necesita validacion con datos locales antes de tomar decisiones nutricionales.
