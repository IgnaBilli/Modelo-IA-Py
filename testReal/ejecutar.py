import json
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
INPUT_PATH = TEST_DIR / "input.json"
OUTPUT_PATH = TEST_DIR / "output.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predictor import HerdPredictor


def main() -> None:
    try:
        contenido = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"No se encontro el archivo de entrada: {INPUT_PATH}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"El archivo input.json no es un JSON valido: {error}")

    animales = contenido.get("animales")
    if not isinstance(animales, list) or not animales:
        raise SystemExit("input.json debe contener una lista no vacia llamada 'animales'.")

    try:
        predictor = HerdPredictor()
        predicciones, advertencias = predictor.predict(animales)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"No se pudo ejecutar la prediccion: {error}")

    salida = {
        "predicciones": predicciones,
        "advertencias": advertencias,
    }
    OUTPUT_PATH.write_text(
        json.dumps(salida, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prediccion completada. Resultado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
