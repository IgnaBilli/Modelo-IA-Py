from src.model.features import CATEGORIES
from src.model.predictor import HerdPredictor

CONDICIONES_CORPORALES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
ESTADOS_FISIOLOGICOS = ("B", "B/L", "L", "N", "N/P", "P", "P/L")


def pedir_texto(mensaje: str, opcional: bool = False) -> str | None:
    while True:
        valor = input(mensaje).strip()
        if valor:
            return valor
        if opcional:
            return None
        print("El valor no puede quedar vacio.")


def pedir_numero(mensaje: str, opcional: bool = False) -> float | None:
    while True:
        valor = input(mensaje).strip()
        if not valor and opcional:
            return None
        try:
            numero = float(valor.replace(",", "."))
            if numero <= 0:
                raise ValueError
            return numero
        except ValueError:
            print("Ingrese un numero mayor que cero (por ejemplo, 420 o 3,5).")


def pedir_categoria() -> str:
    print("\nCategorias disponibles:")
    for indice, categoria in enumerate(CATEGORIES, start=1):
        print(f"  {indice}. {categoria}")

    while True:
        opcion = input("Seleccione una categoria: ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(CATEGORIES):
            return CATEGORIES[int(opcion) - 1]
        print(f"Ingrese un numero entre 1 y {len(CATEGORIES)}.")


def pedir_condicion_corporal() -> float | None:
    opciones = ", ".join(f"{valor:g}" for valor in CONDICIONES_CORPORALES)
    while True:
        condicion = pedir_numero(
            f"Condicion corporal ({opciones}; Enter si no se conoce): ",
            opcional=True,
        )
        if condicion is None or condicion in CONDICIONES_CORPORALES:
            return condicion
        print("Seleccione un valor entre 1.5 y 4.5, en pasos de 0.5.")


def pedir_estado_fisiologico() -> str | None:
    opciones = ", ".join(ESTADOS_FISIOLOGICOS)
    while True:
        estado = pedir_texto(
            f"Estado fisiologico ({opciones}; Enter si no se conoce): ",
            opcional=True,
        )
        if estado is None:
            return None
        estado = estado.upper().replace(" ", "")
        if estado in ESTADOS_FISIOLOGICOS:
            return estado
        print(f"Ingrese una de estas opciones: {opciones}.")


def pedir_animal() -> dict:
    animal_id = pedir_texto("ID del animal: ")
    peso = pedir_numero("Peso vivo (kg): ")
    categoria = pedir_categoria()

    condicion = None
    estado = None
    if categoria == "Vaca":
        condicion = pedir_condicion_corporal()
        estado = pedir_estado_fisiologico()

    return {
        "animal_id": animal_id,
        "peso_vivo_kg": peso,
        "categoria_terrabovina": categoria,
        "condicion_corporal": condicion,
        "estado_fisiologico_norm": estado,
    }


def quiere_agregar_otro() -> bool:
    while True:
        respuesta = input("\nAgregar otro animal? [s/N]: ").strip().lower()
        if respuesta in {"", "n", "no"}:
            return False
        if respuesta in {"s", "si"}:
            return True
        print("Responda 's' para continuar o 'n' para calcular.")


def main() -> None:
    print("=== Prueba del modelo DMI TerraBovina ===\n")
    try:
        predictor = HerdPredictor()
    except FileNotFoundError as error:
        print(error)
        return

    animales = []
    while True:
        print(f"Animal {len(animales) + 1}")
        animales.append(pedir_animal())
        if not quiere_agregar_otro():
            break
        print()

    predicciones, advertencias = predictor.predict(animales)

    print("\n=== Resultados ===")
    for resultado in predicciones:
        print(
            f"{resultado['animal_id']}: {resultado['dmi_kg_dia']:.2f} kg MS/dia "
            f"({resultado['dmi_pct_peso_vivo']:.2f}% del peso vivo)"
        )

    if advertencias:
        print("\nAdvertencias:")
        for advertencia in advertencias:
            print(f"- {advertencia}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nPrueba cancelada.")
