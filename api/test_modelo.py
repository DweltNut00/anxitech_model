from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# Intenta cargar el modelo desde la raiz del proyecto
ruta = BASE_DIR / "modelos" / "modelo_ansiedad.pkl"

print(f"Verificando ruta: {ruta}")
print(f"¿Existe? {ruta.exists()}")
print(f"¿Es archivo? {ruta.is_file()}")

if ruta.exists():
    print("\n✅ Archivo encontrado")
    print(f"Tamaño: {ruta.stat().st_size / 1024:.2f} KB")
    
    # Intentar cargar
    try:
        modelo = joblib.load(str(ruta))
        print(f"✅ Modelo cargado exitosamente")
        print(f"Tipo: {type(modelo)}")
    except Exception as e:
        print(f"❌ Error al cargar: {e}")
else:
    print("\n❌ Archivo NO encontrado")
    print("\nArchivos en el directorio:")
    parent = ruta.parent
    if parent.exists():
        for archivo in parent.iterdir():
            print(f"  - {archivo.name}")
