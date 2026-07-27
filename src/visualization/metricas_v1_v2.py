import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT, REPORT_PATH


V2_REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics_v2_mejorado.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures_metricas_detalladas"
CATEGORIES = ["Vaca", "Vaquilla", "Novillo", "Toro", "Ternero/Ternera"]


def load_metrics():
    v1_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    v2_report = json.loads(V2_REPORT_PATH.read_text(encoding="utf-8"))
    return (
        v1_report["metrics"]["ridge"],
        v2_report["development_oof_metrics"],
        v2_report["nested_oof_metrics"],
    )


def save_global(v1, development, nested):
    rows = []
    for name, metrics in (("V1 Ridge", v1), ("V2 mejorado desarrollo", development), ("V2 mejorado anidado", nested)):
        for metric in ("mae", "rmse", "r2"):
            rows.append({"modelo": name, "metrica": metric.upper(), "valor": metrics[metric]})
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for metric, ax in zip(("MAE", "RMSE", "R2"), axes):
        subset = frame[frame["metrica"] == metric]
        sns.barplot(data=subset, x="modelo", y="valor", hue="modelo", legend=False, ax=ax)
        ax.set_title(f"{metric}: {'menor' if metric != 'R2' else 'mayor'} es mejor", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
        for index, value in enumerate(subset["valor"]):
            ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Comparación global: V1 y V2 mejorado", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_global_tres_evaluaciones.png", dpi=300)
    plt.close()


def save_category_metric(v1, development, nested, metric, filename):
    rows = []
    for category in CATEGORIES:
        for name, source in (("V1 Ridge", v1), ("V2 mejorado desarrollo", development), ("V2 mejorado anidado", nested)):
            rows.append({
                "categoria": category,
                "evaluacion": name,
                "valor": source["por_categoria"][category][metric],
            })
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    sns.barplot(data=frame, x="categoria", y="valor", hue="evaluacion", ax=ax)
    direction = "menor es mejor" if metric != "r2" else "mayor es mejor"
    ax.set_title(f"{metric.upper()} por categoría ({direction})", fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Evaluación")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def save_fold_mae(v1, nested):
    rows = []
    for fold, metrics in enumerate(v1["folds"], start=1):
        rows.append({"fold": str(fold), "modelo": "V1 Ridge", "mae": metrics["mae"]})
    for metrics in nested["folds"]:
        rows.append({"fold": str(metrics["fold"]), "modelo": "V2 mejorado anidado", "mae": metrics["mae"]})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(data=frame, x="fold", y="mae", hue="modelo", ax=ax)
    ax.set_title("MAE por fold externo", fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (kg MS/día)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_mae_por_fold.png", dpi=300)
    plt.close()


def save_percentage_change(v1, development, nested):
    rows = []
    for category in CATEGORIES:
        baseline = v1["por_categoria"][category]["mae"]
        rows.extend([
            {
                "categoria": category,
                "evaluacion": "V2 mejorado desarrollo",
                "cambio": (development["por_categoria"][category]["mae"] - baseline) / baseline * 100,
            },
            {
                "categoria": category,
                "evaluacion": "V2 mejorado anidado",
                "cambio": (nested["por_categoria"][category]["mae"] - baseline) / baseline * 100,
            },
        ])
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=frame, x="categoria", y="cambio", hue="evaluacion", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Cambio porcentual del MAE respecto de V1", fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Cambio MAE (%) - negativo es mejora")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "6_cambio_porcentual_mae.png", dpi=300)
    plt.close()


def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})
    v1, development, nested = load_metrics()
    save_global(v1, development, nested)
    save_category_metric(v1, development, nested, "mae", "2_mae_categoria.png")
    save_category_metric(v1, development, nested, "rmse", "3_rmse_categoria.png")
    save_category_metric(v1, development, nested, "r2", "4_r2_categoria.png")
    save_fold_mae(v1, nested)
    save_percentage_change(v1, development, nested)


if __name__ == "__main__":
    generate()
