# setup.py - Instalar librerías necesarias
import subprocess
import sys

def install_libraries():
    libraries = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'mysql-connector-python',
        'openpyxl'
    ]
    
    print("=== INSTALANDO LIBRERÍAS ===")
    for lib in libraries:
        print(f"Instalando {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"✅ {lib} instalado")
    
    print("\n🎉 ¡Todas las librerías instaladas!")

if __name__ == "__main__":
    install_libraries()