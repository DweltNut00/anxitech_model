"""
COMPARACIÓN DE MODELOS: ¿Cuál es el bueno?
==========================================
Compara los dos modelos que tienes y te dice cuál usar

Autor: Sistema AnxiTech
Fecha: 2026-02-05
"""

import joblib
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

print("="*80)
print("COMPARACIÓN DE MODELOS: notebooks/modelos vs modelos/")
print("="*80)

# ============================================
# DEFINIR RUTAS DE LOS MODELOS
# ============================================

# Modelo 1: En carpeta notebooks/modelos/
modelo1_path = BASE_DIR / "notebooks" / "modelos" / "modelo_ansiedad.pkl"
metadata1_path = BASE_DIR / "notebooks" / "modelos" / "metadata.json"

# Modelo 2: En carpeta raíz modelos/
modelo2_path = BASE_DIR / "modelos" / "modelo_ansiedad.pkl"
metadata2_path = BASE_DIR / "modelos" / "metadata.json"

print("\n📂 UBICACIONES:")
print(f"   Modelo 1: {modelo1_path.absolute()}")
print(f"   Modelo 2: {modelo2_path.absolute()}")

# ============================================
# FUNCIÓN PARA ANALIZAR UN MODELO
# ============================================

def analizar_modelo(modelo_path, metadata_path, nombre):
    """Analiza un modelo y retorna sus características"""
    
    print(f"\n{'='*80}")
    print(f"ANALIZANDO: {nombre}")
    print(f"{'='*80}\n")
    
    # Verificar si existe
    if not modelo_path.exists():
        print(f"   ❌ ARCHIVO NO ENCONTRADO")
        print(f"      Ruta: {modelo_path.absolute()}")
        return None
    
    print(f"   ✅ Archivo encontrado")
    
    # Info del archivo
    stat = modelo_path.stat()
    fecha_mod = datetime.fromtimestamp(stat.st_mtime)
    tamaño_mb = stat.st_size / (1024 * 1024)
    
    print(f"   📅 Última modificación: {fecha_mod}")
    print(f"   📦 Tamaño: {tamaño_mb:.2f} MB")
    
    # Cargar modelo
    try:
        modelo = joblib.load(modelo_path)
        print(f"   ✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"   ❌ Error al cargar: {e}")
        return None
    
    # Características del modelo
    print(f"\n   🔍 CARACTERÍSTICAS DEL MODELO:")
    print(f"      Tipo: {type(modelo).__name__}")
    
    if hasattr(modelo, 'n_features_in_'):
        n_features = modelo.n_features_in_
        print(f"      Número de features: {n_features}")
    else:
        n_features = None
        print(f"      ⚠️  No se puede determinar número de features")
    
    if hasattr(modelo, 'n_estimators'):
        print(f"      Número de árboles: {modelo.n_estimators}")
    
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        print(f"      Feature importances suma: {importances.sum():.6f}")
        
        # Top 3 importances
        feature_names = [
            'promedio_anterior', 'semestre', 'materias', 'edad',
            'transporte', 'familiares', 'trabajo', 'beca',
            'sexo', 'estado_civil', 'carrera'
        ]
        
        if len(importances) == len(feature_names):
            print(f"\n      🏆 TOP 3 FACTORES:")
            sorted_features = sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(sorted_features[:3], 1):
                print(f"         {i}. {feat:<20} {imp*100:.2f}%")
        else:
            print(f"      ⚠️  Número de features no coincide: {len(importances)} vs {len(feature_names)}")
    
    # Leer metadata si existe
    if metadata_path.exists():
        print(f"\n   📄 METADATA:")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"      Fecha entrenamiento: {metadata.get('fecha_entrenamiento', 'N/A')}")
            print(f"      Accuracy test: {metadata.get('accuracy_test', 'N/A')}")
            print(f"      N° registros: {metadata.get('n_registros', 'N/A')}")
            
            if 'features' in metadata:
                print(f"      Features en metadata: {len(metadata['features'])}")
        except Exception as e:
            print(f"      ⚠️  Error al leer metadata: {e}")
            metadata = None
    else:
        print(f"\n   ⚠️  No existe archivo metadata.json")
        metadata = None
    
    return {
        'existe': True,
        'fecha_modificacion': fecha_mod,
        'tamaño_mb': tamaño_mb,
        'n_features': n_features,
        'modelo': modelo,
        'metadata': metadata,
        'importances': modelo.feature_importances_ if hasattr(modelo, 'feature_importances_') else None
    }

# ============================================
# ANALIZAR AMBOS MODELOS
# ============================================

resultado1 = analizar_modelo(modelo1_path, metadata1_path, "MODELO 1 (notebooks/modelos/)")
resultado2 = analizar_modelo(modelo2_path, metadata2_path, "MODELO 2 (modelos/)")

# ============================================
# COMPARACIÓN Y RECOMENDACIÓN
# ============================================

print("\n" + "="*80)
print("🎯 COMPARACIÓN Y RECOMENDACIÓN")
print("="*80 + "\n")

if resultado1 is None and resultado2 is None:
    print("❌ ERROR: Ninguno de los dos modelos existe")
    print("\n   Solución: Ejecuta python entrenar_modelo_basico.py")
    exit()

elif resultado1 is None:
    print("✅ RECOMENDACIÓN: Usar MODELO 2 (modelos/)")
    print("   Razón: Es el único que existe")
    modelo_recomendado = "modelos/modelo_ansiedad.pkl"
    
elif resultado2 is None:
    print("✅ RECOMENDACIÓN: Usar MODELO 1 (notebooks/modelos/)")
    print("   Razón: Es el único que existe")
    modelo_recomendado = "notebooks/modelos/modelo_ansiedad.pkl"
    
else:
    # Ambos existen, comparar
    print("📊 COMPARACIÓN DETALLADA:\n")
    
    print(f"{'Característica':<30} {'Modelo 1 (notebooks)':<25} {'Modelo 2 (raíz)':<25}")
    print("-"*80)
    print(f"{'Fecha modificación':<30} {str(resultado1['fecha_modificacion']):<25} {str(resultado2['fecha_modificacion']):<25}")
    print(f"{'Tamaño (MB)':<30} {resultado1['tamaño_mb']:<25.2f} {resultado2['tamaño_mb']:<25.2f}")
    print(f"{'Número de features':<30} {str(resultado1['n_features']):<25} {str(resultado2['n_features']):<25}")
    
    # Determinar cuál es el correcto
    puntos_modelo1 = 0
    puntos_modelo2 = 0
    
    # Criterio 1: Número de features (debe ser 11)
    if resultado1['n_features'] == 11:
        puntos_modelo1 += 3
        print(f"{'Features correctos (11)':<30} {'✅ SÍ':<25} {'❌ NO' if resultado2['n_features'] != 11 else '✅ SÍ':<25}")
    elif resultado2['n_features'] == 11:
        puntos_modelo2 += 3
        print(f"{'Features correctos (11)':<30} {'❌ NO':<25} {'✅ SÍ':<25}")
    
    # Criterio 2: Más reciente
    if resultado1['fecha_modificacion'] > resultado2['fecha_modificacion']:
        puntos_modelo1 += 2
        print(f"{'Más reciente':<30} {'✅ SÍ':<25} {'❌ NO':<25}")
    else:
        puntos_modelo2 += 2
        print(f"{'Más reciente':<30} {'❌ NO':<25} {'✅ SÍ':<25}")
    
    # Criterio 3: Tiene metadata
    if resultado1['metadata'] is not None:
        puntos_modelo1 += 1
    if resultado2['metadata'] is not None:
        puntos_modelo2 += 1
    
    print(f"\n{'PUNTUACIÓN TOTAL':<30} {puntos_modelo1:<25} {puntos_modelo2:<25}")
    
    # Decisión final
    print("\n" + "="*80)
    if puntos_modelo1 > puntos_modelo2:
        print("✅ RECOMENDACIÓN: Usar MODELO 1 (notebooks/modelos/)")
        print(f"   Razón: Mejor puntuación ({puntos_modelo1} vs {puntos_modelo2})")
        modelo_recomendado = "notebooks/modelos/modelo_ansiedad.pkl"
    elif puntos_modelo2 > puntos_modelo1:
        print("✅ RECOMENDACIÓN: Usar MODELO 2 (modelos/)")
        print(f"   Razón: Mejor puntuación ({puntos_modelo2} vs {puntos_modelo1})")
        modelo_recomendado = "modelos/modelo_ansiedad.pkl"
    else:
        # Empate, usar el más reciente
        if resultado1['fecha_modificacion'] > resultado2['fecha_modificacion']:
            print("✅ RECOMENDACIÓN: Usar MODELO 1 (notebooks/modelos/)")
            print("   Razón: Es más reciente")
            modelo_recomendado = "notebooks/modelos/modelo_ansiedad.pkl"
        else:
            print("✅ RECOMENDACIÓN: Usar MODELO 2 (modelos/)")
            print("   Razón: Es más reciente")
            modelo_recomendado = "modelos/modelo_ansiedad.pkl"

# ============================================
# ACCIONES A TOMAR
# ============================================

print("\n" + "="*80)
print("🔧 ACCIONES A TOMAR")
print("="*80 + "\n")

print(f"1️⃣  ACTUALIZAR analytics.py")
print(f"   Abre analytics.py y busca la línea:")
print(f"   modelo_path = Path(r'...')")
print(f"\n   Cámbiala por:")
print(f"   modelo_path = BASE_DIR / '{Path(modelo_recomendado).as_posix()}'")

print(f"\n2️⃣  VERIFICAR QUE EL MODELO TIENE 11 FEATURES")
if resultado1 and resultado1['n_features'] != 11:
    print(f"   ⚠️  Modelo 1 tiene {resultado1['n_features']} features (debería ser 11)")
if resultado2 and resultado2['n_features'] != 11:
    print(f"   ⚠️  Modelo 2 tiene {resultado2['n_features']} features (debería ser 11)")

print(f"\n3️⃣  ACTUALIZAR feature_names en analytics.py")
print(f"   Asegúrate de que tenga 11 features:")
print(f"   self.feature_names = [")
print(f"       'promedio_anterior', 'semestre', 'materias', 'edad',")
print(f"       'transporte', 'familiares', 'trabajo', 'beca',")
print(f"       'sexo', 'estado_civil', 'carrera'")
print(f"   ]")

print(f"\n4️⃣  REINICIAR LA API")
print(f"   python main.py")

print(f"\n5️⃣  VERIFICAR QUE FUNCIONÓ")
print(f"   python comparar_modelo_dashboard.py")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
