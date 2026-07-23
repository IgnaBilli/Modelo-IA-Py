import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})


def generate_all_plots(
    y_true: pd.Series,
    y_pred: np.ndarray,
    X_df: pd.DataFrame,
    metrics_report: dict,
    winning_model,
    output_dir: Path,
) -> list[Path]:
    """
    Generate complete suite of 8 analytical graphics and save high-resolution figures.
    Returns list of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    residuals = y_true.values - y_pred

    logger.info(f"Generando suite completa de gráficos en: {output_dir}")

    # 1. Model Comparison (MAE & R2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [m for m in metrics_report.keys()]
    maes = [metrics_report[m]["mae"] for m in model_names]
    r2s = [metrics_report[m]["r2"] for m in model_names]

    sns.barplot(x=model_names, y=maes, ax=ax1, palette="Blues_d", hue=model_names, legend=False)
    ax1.set_title("Comparación de MAE entre Modelos (Menor es mejor)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAE (kg MS/día)")
    for i, v in enumerate(maes):
        ax1.text(i, v + 0.05, f"{v:.3f}", ha="center", fontweight="bold")

    sns.barplot(x=model_names, y=r2s, ax=ax2, palette="Greens_d", hue=model_names, legend=False)
    ax2.set_title("Comparación de R² entre Modelos (Mayor es mejor)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("R² Score")
    for i, v in enumerate(r2s):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    p1 = output_dir / "1_model_comparison_mae_r2.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    generated_files.append(p1)

    # 2. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#1f77b4", edgecolor=None, ax=ax)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Línea de Identidad (y = x)")

    winner_name = [k for k in metrics_report.keys() if metrics_report[k].get("mae") == min(maes)][0]
    r2_val = metrics_report[winner_name]["r2"]
    mae_val = metrics_report[winner_name]["mae"]
    ax.text(
        0.05,
        0.90,
        f"Modelo: {winner_name}\nR² = {r2_val:.3f}\nMAE = {mae_val:.3f} kg/día",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        fontsize=11,
    )
    ax.set_title("DMI Real vs DMI Predicho", fontsize=14, fontweight="bold")
    ax.set_xlabel("DMI Real (kg MS/día)", fontsize=12)
    ax.set_ylabel("DMI Predicho (kg MS/día)", fontsize=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p2 = output_dir / "2_actual_vs_predicted.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    generated_files.append(p2)

    # 3. Residuals Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax, bins=30)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error Cero")
    ax.set_title("Distribución de Residuos (Errores: Real - Predicho)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Residual (kg MS/día)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    plt.tight_layout()
    p3 = output_dir / "3_residuals_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    generated_files.append(p3)

    # 4. Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5))
    feature_names = []
    importances = []

    estimator = winning_model
    if isinstance(winning_model, Pipeline):
        preprocessor = winning_model.named_steps.get("preprocessor")
        estimator = winning_model.named_steps.get("model")
        if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = []

    if not feature_names:
        feature_names = list(X_df.columns)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "get_feature_importance"):
        importances = estimator.get_feature_importance()
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_)
        if importances.ndim > 1:
            importances = importances.ravel()

    if len(importances) > 0 and len(feature_names) == len(importances):
        clean_names = [
            name.replace("numeric__", "")
            .replace("categorical__", "")
            .replace("categoria_terrabovina_", "Cat: ")
            .replace("estado_fisiologico_norm_", "Estado: ")
            for name in feature_names
        ]
        fi_df = pd.DataFrame({"feature": clean_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).head(10)

        sns.barplot(data=fi_df, x="importance", y="feature", palette="cividis", hue="feature", legend=False, ax=ax)
        ax.set_title("Importancia / Peso de Variables (Feature Importance)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Magnitud / Importancia Absoluta")
        ax.set_ylabel("Variable / Atributo")
        max_val = fi_df["importance"].max() if len(fi_df) > 0 else 1.0
        for i, v in enumerate(fi_df["importance"]):
            ax.text(v + 0.01 * max_val, i, f"{v:.3f}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Importancia no disponible para este estimador", ha="center", va="center")

    plt.tight_layout()
    p4 = output_dir / "4_feature_importance.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    generated_files.append(p4)

    # 5. MAE by Category
    fig, ax = plt.subplots(figsize=(9, 5))
    winner_cat_metrics = metrics_report[winner_name].get("por_categoria", {})
    if winner_cat_metrics:
        cats = list(winner_cat_metrics.keys())
        cat_maes = [winner_cat_metrics[c]["mae"] for c in cats]
        sns.barplot(x=cats, y=cat_maes, palette="rocket", hue=cats, legend=False, ax=ax)
        ax.set_title("Error MAE del Modelo Ganador por Categoría de Animal", fontsize=13, fontweight="bold")
        ax.set_xlabel("Categoría TerraBovina")
        ax.set_ylabel("MAE (kg MS/día)")
        for i, v in enumerate(cat_maes):
            ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    p5 = output_dir / "5_mae_by_category.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    generated_files.append(p5)

    # 6. Weight vs DMI Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    temp_df = X_df.copy()
    temp_df["dmi_real"] = y_true.values
    sns.scatterplot(
        data=temp_df,
        x="peso_vivo_kg",
        y="dmi_real",
        hue="categoria_terrabovina",
        alpha=0.7,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("Relación Peso Vivo (kg) vs Consumo DMI Real por Categoría", fontsize=13, fontweight="bold")
    ax.set_xlabel("Peso Vivo (kg)")
    ax.set_ylabel("DMI Real (kg MS/día)")
    ax.legend(title="Categoría", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    p6 = output_dir / "6_weight_vs_dmi_scatter.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    generated_files.append(p6)

    # 7. DMI by Category Boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=temp_df, x="categoria_terrabovina", y="dmi_real", palette="Set2", hue="categoria_terrabovina", legend=False, ax=ax)
    ax.set_title("Distribución de Consumo DMI Real por Categoría de Animal", fontsize=13, fontweight="bold")
    ax.set_xlabel("Categoría TerraBovina")
    ax.set_ylabel("DMI Real (kg MS/día)")
    plt.tight_layout()
    p7 = output_dir / "7_dmi_by_category_boxplot.png"
    plt.savefig(p7, dpi=300)
    plt.close()
    generated_files.append(p7)

    # 8. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
    corr = temp_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Mapa de Calor de Correlación entre Variables Numéricas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p8 = output_dir / "8_correlation_heatmap.png"
    plt.savefig(p8, dpi=300)
    plt.close()
    generated_files.append(p8)

    logger.info(f"Se generaron exitosamente los {len(generated_files)} gráficos en {output_dir}.")
    return generated_files
