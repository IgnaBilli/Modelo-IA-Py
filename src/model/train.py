import json
from datetime import datetime, timezone

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from src.config import DATA_PATH, MODEL_PATH, REPORT_PATH
from src.model.features import CATEGORICAL_FEATURES, CATEGORIES, FEATURES, NUMERIC_FEATURES, load_training_data

RANDOM_STATE = 42


def sklearn_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def model_candidates() -> dict:
    return {
        "baseline_media": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([
            ("preprocessor", sklearn_preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            cat_features=CATEGORICAL_FEATURES,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def metric_summary(y_true, predictions) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_true, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }


def train() -> dict:
    X, y, groups = load_training_data(DATA_PATH)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = model_candidates()
    report = {}

    total_fits = len(candidates) * splitter.get_n_splits() + 1
    with tqdm(total=total_fits, desc="Entrenando", unit="ajuste", dynamic_ncols=True) as progress:
        for name in candidates:
            out_of_fold = np.zeros(len(y), dtype=float)
            fold_metrics = []
            splits = splitter.split(X, X["categoria_terrabovina"], groups)
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                progress.set_postfix_str(f"modelo={name}, fold={fold}/5")
                if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
                    raise RuntimeError("Fuga de hogares detectada en validacion")
                model = model_candidates()[name]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                predictions = np.maximum(model.predict(X.iloc[test_idx]), 0.0)
                out_of_fold[test_idx] = predictions
                fold_metrics.append(metric_summary(y.iloc[test_idx], predictions))
                progress.update()

            report[name] = metric_summary(y, out_of_fold)
            report[name]["folds"] = fold_metrics
            report[name]["por_categoria"] = {
                category: metric_summary(y[X["categoria_terrabovina"] == category],
                                         out_of_fold[X["categoria_terrabovina"] == category])
                for category in CATEGORIES
            }

        winner = min(report, key=lambda name: report[name]["mae"])
        progress.set_postfix_str(f"modelo={winner}, entrenamiento final")
        final_model = candidates[winner]
        final_model.fit(X, y)
        progress.update()

    weight_ranges = {
        category: {
            "min": round(float(X.loc[X["categoria_terrabovina"] == category, "peso_vivo_kg"].min()), 1),
            "max": round(float(X.loc[X["categoria_terrabovina"] == category, "peso_vivo_kg"].max()), 1),
        }
        for category in CATEGORIES
    }
    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(X),
        "dataset": DATA_PATH.name,
        "dataset_source": "terrabovina_dmi_kenya_features V2.csv",
        "features": FEATURES,
        "categories": CATEGORIES,
        "winner": winner,
        "metrics": report,
        "weight_ranges_kg": weight_ranges,
        "limitations": [
            "Entrenado con datos de sistemas productivos de Bomet, Kenia.",
            "No incluye clima, calidad o disponibilidad del forraje.",
            "Debe validarse con datos locales antes de usarse para decisiones nutricionales en Argentina.",
        ],
    }
    bundle = {"model": final_model, "metadata": metadata}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"winner": winner, "metrics": report[winner]}, indent=2))
    return metadata


if __name__ == "__main__":
    train()
