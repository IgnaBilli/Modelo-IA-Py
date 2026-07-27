from datetime import datetime, timezone
import json

from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from src.config import DATA_PATH, PROJECT_ROOT, REPORT_PATH
from src.model.category_blend import CategoryBlendRegressor, add_physiological_features
from src.model.features import CATEGORICAL_FEATURES, CATEGORIES, load_training_data


RANDOM_STATE = 42
BASELINE_NAME = "ridge_v1"
MODEL_PATH = PROJECT_ROOT / "models" / "dmi_model_v2_mejorado.joblib"
REPORT_PATH_V2 = PROJECT_ROOT / "reports" / "model_metrics_v2_mejorado.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures_v2_mejorado"

NUMERIC_FEATURES = [
    "peso_vivo_kg",
    "peso_vivo_kg_sq",
    "peso_vivo_kg_sqrt",
    "condicion_corporal",
    "peso_x_condicion",
    "esta_lactando",
    "esta_preniada",
    "esta_seca",
    "estado_normal",
    "estado_desconocido",
    "peso_x_lactando",
    "peso_x_preniada",
    *[f"peso_categoria_{category}" for category in CATEGORIES],
]

SPECS = [
    {"name": BASELINE_NAME, "family": "ridge_v1", "alpha": 10.0},
    {"name": "ridge_fisio_a0.3", "family": "ridge_fisio", "alpha": 0.3},
    {"name": "ridge_fisio_a3", "family": "ridge_fisio", "alpha": 3.0},
    {"name": "ridge_fisio_a10", "family": "ridge_fisio", "alpha": 10.0},
    {"name": "ridge_fisio_a100", "family": "ridge_fisio", "alpha": 100.0},
    {"name": "catboost_rmse", "family": "catboost", "loss": "RMSE"},
    {"name": "catboost_mae", "family": "catboost", "loss": "MAE"},
]


def preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def build_model(spec: dict):
    if spec["family"] == "ridge_v1":
        return Pipeline([
            ("preprocessor", preprocessor(["peso_vivo_kg", "condicion_corporal"])),
            ("model", Ridge(alpha=spec["alpha"])),
        ])
    if spec["family"] == "ridge_fisio":
        return Pipeline([
            ("feature_engineering", FunctionTransformer(add_physiological_features, validate=False)),
            ("preprocessor", preprocessor(NUMERIC_FEATURES)),
            ("model", Ridge(alpha=spec["alpha"])),
        ])
    return CatBoostRegressor(
        iterations=350,
        depth=6,
        learning_rate=0.045,
        loss_function=spec["loss"],
        cat_features=CATEGORICAL_FEATURES,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_true, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
        "bias_real_minus_pred": round(float(np.mean(np.asarray(y_true) - predictions)), 4),
    }


def full_metrics(X: pd.DataFrame, y: pd.Series, predictions: np.ndarray) -> dict:
    result = metrics(y, predictions)
    result["por_categoria"] = {
        category: metrics(
            y[X["categoria_terrabovina"] == category],
            predictions[X["categoria_terrabovina"] == category],
        )
        for category in CATEGORIES
    }
    result["macro_mae"] = round(float(np.mean([
        result["por_categoria"][category]["mae"] for category in CATEGORIES
    ])), 4)
    result["worst_category_mae"] = round(float(max(
        result["por_categoria"][category]["mae"] for category in CATEGORIES
    )), 4)
    return result


def oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> dict[str, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, X["categoria_terrabovina"], groups))
    predictions = {spec["name"]: np.zeros(len(y), dtype=float) for spec in SPECS}
    for spec in SPECS:
        for train_idx, test_idx in splits:
            model = build_model(spec)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions[spec["name"]][test_idx] = np.maximum(model.predict(X.iloc[test_idx]), 0.0)
    return predictions


def choose_routing(X: pd.DataFrame, y: pd.Series, predictions: dict[str, np.ndarray]) -> dict:
    routing = {}
    categories = X["categoria_terrabovina"].to_numpy()
    baseline = predictions[BASELINE_NAME]
    weights = np.linspace(0.0, 1.0, 21)

    for category in CATEGORIES:
        mask = categories == category
        y_category = y[mask]
        baseline_mae = mean_absolute_error(y_category, baseline[mask])
        baseline_rmse = mean_squared_error(y_category, baseline[mask]) ** 0.5
        best = {
            "specialist": BASELINE_NAME,
            "specialist_weight": 0.0,
            "score": 1.0,
            "inner_mae": baseline_mae,
            "inner_rmse": baseline_rmse,
        }
        for spec in SPECS:
            if spec["name"] == BASELINE_NAME:
                continue
            for weight in weights[1:]:
                blended = (1.0 - weight) * baseline[mask] + weight * predictions[spec["name"]][mask]
                mae = mean_absolute_error(y_category, blended)
                rmse = mean_squared_error(y_category, blended) ** 0.5
                # The max term prevents trading a large RMSE loss for a small MAE gain.
                mae_ratio = mae / baseline_mae
                rmse_ratio = rmse / baseline_rmse
                score = 0.45 * mae_ratio + 0.35 * rmse_ratio + 0.20 * max(mae_ratio, rmse_ratio)
                if score < best["score"]:
                    best = {
                        "specialist": spec["name"],
                        "specialist_weight": round(float(weight), 2),
                        "score": round(float(score), 6),
                        "inner_mae": round(float(mae), 4),
                        "inner_rmse": round(float(rmse), 4),
                    }
        routing[category] = best
    return routing


def blend_oof_predictions(
    X: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    routing: dict,
) -> np.ndarray:
    blended = predictions[BASELINE_NAME].copy()
    categories = X["categoria_terrabovina"].to_numpy()
    for category, route in routing.items():
        mask = categories == category
        weight = float(route["specialist_weight"])
        specialist = predictions[route["specialist"]]
        blended[mask] = (1.0 - weight) * blended[mask] + weight * specialist[mask]
    return np.maximum(blended, 0.0)


def fit_models(X: pd.DataFrame, y: pd.Series, names: set[str]) -> dict:
    models = {}
    for spec in SPECS:
        if spec["name"] in names:
            model = build_model(spec)
            model.fit(X, y)
            models[spec["name"]] = model
    return models


def predict_with_routing(models: dict, routing: dict, X: pd.DataFrame) -> np.ndarray:
    return CategoryBlendRegressor(models, routing, BASELINE_NAME).predict(X)


def nested_evaluation(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, list[dict]]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    nested_predictions = np.zeros(len(y), dtype=float)
    fold_details = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, X["categoria_terrabovina"], groups), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].reset_index(drop=True)
        inner_predictions = oof_predictions(X_train, y_train, groups_train, n_splits=3)
        routing = choose_routing(X_train, y_train, inner_predictions)
        selected_names = {BASELINE_NAME, *[route["specialist"] for route in routing.values()]}
        models = fit_models(X.iloc[train_idx], y.iloc[train_idx], selected_names)
        fold_predictions = predict_with_routing(models, routing, X.iloc[test_idx])
        nested_predictions[test_idx] = fold_predictions
        detail = metrics(y.iloc[test_idx], fold_predictions)
        detail.update({"fold": fold, "routing": routing})
        fold_details.append(detail)
    return nested_predictions, fold_details


def comparison(v1: dict, v2: dict) -> dict:
    result = {
        metric: {
            "v1": v1[metric],
            "v2": v2[metric],
            "difference_v2_minus_v1": round(v2[metric] - v1[metric], 4),
        }
        for metric in ("mae", "rmse", "r2")
    }
    result["por_categoria"] = {
        category: {
            metric: {
                "v1": v1["por_categoria"][category][metric],
                "v2": v2["por_categoria"][category][metric],
                "difference_v2_minus_v1": round(
                    v2["por_categoria"][category][metric] - v1["por_categoria"][category][metric],
                    4,
                ),
            }
            for metric in ("mae", "rmse", "r2")
        }
        for category in CATEGORIES
    }
    return result


def generate_plots(v1: dict, v2: dict, X: pd.DataFrame, y: pd.Series, predictions: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    global_df = pd.DataFrame([
        {"modelo": "V1", "MAE": v1["mae"], "RMSE": v1["rmse"], "R2": v1["r2"]},
        {"modelo": "V2 mejorado", "MAE": v2["mae"], "RMSE": v2["rmse"], "R2": v2["r2"]},
    ])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for metric, ax in zip(("MAE", "RMSE", "R2"), axes):
        sns.barplot(data=global_df, x="modelo", y=metric, hue="modelo", legend=False, ax=ax)
        ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "1_global_comparison.png", dpi=300)
    plt.close()

    rows = []
    for category in CATEGORIES:
        for model_name, values in (("V1", v1), ("V2 mejorado", v2)):
            row = {"categoria": category, "modelo": model_name}
            row.update(values["por_categoria"][category])
            rows.append(row)
    category_df = pd.DataFrame(rows)
    for index, metric in enumerate(("mae", "rmse", "r2"), start=2):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=category_df, x="categoria", y=metric, hue="modelo", ax=ax)
        ax.set_title(f"{metric.upper()} por categoría")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{index}_{metric}_by_category.png", dpi=300)
        plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=y, y=predictions, hue=X["categoria_terrabovina"], alpha=0.5, ax=ax)
    limits = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
    ax.plot(limits, limits, "r--")
    ax.set_title("V2 mejorado: real vs predicho")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "5_actual_vs_predicted.png", dpi=300)
    plt.close()


def train() -> dict:
    X, y, groups = load_training_data(DATA_PATH)
    nested_predictions, fold_details = nested_evaluation(X, y, groups)
    nested_metrics = full_metrics(X, y, nested_predictions)
    nested_metrics["folds"] = fold_details

    final_oof = oof_predictions(X, y, groups, n_splits=5)
    final_routing = choose_routing(X, y, final_oof)
    development_predictions = blend_oof_predictions(X, final_oof, final_routing)
    development_metrics = full_metrics(X, y, development_predictions)
    selected_names = {BASELINE_NAME, *[route["specialist"] for route in final_routing.values()]}
    final_models = fit_models(X, y, selected_names)
    final_model = CategoryBlendRegressor(final_models, final_routing, BASELINE_NAME)

    v1_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    v1_metrics = v1_report["metrics"][v1_report["winner"]]
    report = {
        "version": "2.1.0-experimental",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(X),
        "model_type": "CategoryBlendRegressor",
        "physiological_states": {
            "L": "Lactando",
            "P": "Preñada / Gestante",
            "B": "Seca / Vacía",
            "N": "Normal / No gestante ni lactante",
            "P/L": "Preñada y Lactando",
            "B/L": "Seca y Lactando / transición",
            "N/P": "No lactante y Preñada",
            "DESCONOCIDO": "Sin registro",
        },
        "validation": "Nested StratifiedGroupKFold: 5 outer and 3 inner folds",
        "final_routing": final_routing,
        "nested_oof_metrics": nested_metrics,
        "development_oof_metrics": development_metrics,
        "comparison_with_v1_nested": comparison(v1_metrics, nested_metrics),
        "comparison_with_v1_development": comparison(v1_metrics, development_metrics),
        "weight_ranges_kg": {
            category: {
                "min": round(float(X.loc[X["categoria_terrabovina"] == category, "peso_vivo_kg"].min()), 1),
                "max": round(float(X.loc[X["categoria_terrabovina"] == category, "peso_vivo_kg"].max()), 1),
            }
            for category in CATEGORIES
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH_V2.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "metadata": report}, MODEL_PATH)
    REPORT_PATH_V2.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_plots(v1_metrics, development_metrics, X, y, development_predictions)
    print(json.dumps({
        "final_routing": final_routing,
        "nested_oof_metrics": nested_metrics,
        "development_oof_metrics": development_metrics,
        "comparison_with_v1_nested": report["comparison_with_v1_nested"],
        "comparison_with_v1_development": report["comparison_with_v1_development"],
    }, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    train()
