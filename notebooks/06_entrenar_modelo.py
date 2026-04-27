"""
ENTRENAMIENTO DE MODELO ML - ANXITECH (VERSIÓN CORREGIDA)
============================================================
Extrae datos DIRECTAMENTE de la BD para evitar problemas con CSVs antiguos

Autor: Sistema AnxiTech
Fecha: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mysql.connector
import warnings
from datetime import datetime
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ENTRENAMIENTO DE MODELO - RANDOM FOREST (DESDE BD)")
print("="*80)

# ============================================
# 1. CREAR CARPETAS
# ============================================
print("\n📁 Creando carpetas...")

dirs = {
    'modelos': Path('modelos'),
    'reportes': Path('reportes'),
    'graficos_ml': Path('graficos_ml')
}

for name, path in dirs.items():
    path.mkdir(exist_ok=True)

print("✅ Carpetas listas")

# ============================================
# 2. EXTRAER DATOS DE LA BD
# ============================================
print("\n" + "="*80)
print("EXTRAYENDO DATOS DE LA BASE DE DATOS")
print("="*80)

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="anxitech"
    )
    
    # Query para extraer datos con 11 variables
    query = """
    SELECT 
        c.promedio_anterior,
        c.semestre,
        c.materias,
        c.edad,
        c.transporte,
        c.familiares,
        c.trabajo,
        c.beca,
        c.sexo,
        c.estado_civil,
        c.carrera,
        SUM(ap.valor) as suma_ansiedad,
        COUNT(ap.id) as num_respuestas
    FROM complemento c
    INNER JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
    INNER JOIN pregunta p ON ap.id_pregunta = p.id
    WHERE p.categoria = 'ansiedad' AND p.status = 1
    GROUP BY c.id_alumno
    HAVING COUNT(ap.id) = 7
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Datos extraídos: {len(df)} registros")
    
    if len(df) == 0:
        print("\n❌ ERROR: No hay datos en la BD")
        print("   Ejecuta primero: python 10_cargar_dataset_hibrido_bd_CORREGIDO.py")
        exit()
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ============================================
# 3. CLASIFICAR NIVELES DE ANSIEDAD
# ============================================
print("\n📊 Clasificando niveles de ansiedad...")

def clasificar_ansiedad(suma):
    """Clasifica según suma DASS-21 (0-21)"""
    if suma <= 4:
        return 'Bajo'
    elif suma <= 7:
        return 'Medio'
    else:
        return 'Alto'

df['nivel_ansiedad'] = df['suma_ansiedad'].apply(clasificar_ansiedad)

print(f"\n   Distribución de suma_ansiedad:")
print(f"   Mínimo: {df['suma_ansiedad'].min()}")
print(f"   Máximo: {df['suma_ansiedad'].max()}")
print(f"   Media: {df['suma_ansiedad'].mean():.2f}")

print(f"\n   Distribución de niveles:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = (df['nivel_ansiedad'] == nivel).sum()
    pct = (count / len(df)) * 100
    emoji = {'Bajo': '🟢', 'Medio': '🟡', 'Alto': '🔴'}[nivel]
    print(f"      {emoji} {nivel}: {count} ({pct:.1f}%)")

# Verificar que hay al menos 2 clases
clases_unicas = df['nivel_ansiedad'].nunique()
if clases_unicas < 2:
    print(f"\n❌ ERROR: Solo hay {clases_unicas} clase(s)")
    print("   El modelo necesita al menos 2 clases diferentes")
    exit()

# ============================================
# 4. PREPARAR DATOS
# ============================================
print("\n" + "="*80)
print("PREPARANDO DATOS PARA ENTRENAMIENTO")
print("="*80)

feature_names = ['promedio_anterior', 'semestre', 'materias', 'edad',
                 'transporte', 'familiares', 'trabajo', 'beca',
                 'sexo', 'estado_civil', 'carrera']

print(f"\n📋 Variables (11):")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

X = df[feature_names].copy()
y = df['nivel_ansiedad'].copy()

print(f"\n✅ Features: {X.shape}")
print(f"✅ Target: {y.shape}")

# ============================================
# 5. CODIFICAR VARIABLES CATEGÓRICAS
# ============================================
print("\n🔤 Codificando variables categóricas...")

le_sexo = LabelEncoder()
le_estado_civil = LabelEncoder()
le_carrera = LabelEncoder()

X['sexo'] = le_sexo.fit_transform(X['sexo'].astype(str))
X['estado_civil'] = le_estado_civil.fit_transform(X['estado_civil'].astype(str))
X['carrera'] = le_carrera.fit_transform(X['carrera'].astype(str))

print(f"\n   ✅ Sexo: {list(le_sexo.classes_)}")
print(f"   ✅ Estado civil: {list(le_estado_civil.classes_)}")
print(f"   ✅ Carrera: {list(le_carrera.classes_)}")

# Verificar valores nulos
if X.isnull().sum().sum() > 0:
    print(f"\n⚠️  Valores nulos detectados")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
    print(f"   ✅ Valores nulos imputados")

# ============================================
# 6. DIVIDIR DATOS
# ============================================
print("\n📊 Dividiendo datos...")

# Verificar que hay suficientes muestras por clase para el split
min_samples = y.value_counts().min()
if min_samples < 10:
    print(f"\n⚠️  ADVERTENCIA: Clase con pocas muestras ({min_samples})")
    print("   Usando split sin stratify para evitar errores")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"\n   ✅ Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ✅ Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Verificar distribución en train y test
print(f"\n   Distribución en TRAIN:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = (y_train == nivel).sum()
    if count > 0:
        print(f"      {nivel}: {count}")

print(f"\n   Distribución en TEST:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = (y_test == nivel).sum()
    if count > 0:
        print(f"      {nivel}: {count}")

# ============================================
# 7. ENTRENAR MODELO
# ============================================
print("\n" + "="*80)
print("🤖 ENTRENANDO RANDOM FOREST")
print("="*80)

modelo = RandomForestClassifier(
    n_estimators=50,         # Reducido para evitar overfitting
    max_depth=5,             # Menos profundidad
    min_samples_split=10,    # Más muestras para dividir
    min_samples_leaf=5,      # Más muestras en hojas
    max_features='sqrt',     # Solo sqrt(n) features por árbol
    random_state=42,
    class_weight='balanced',   # ← agregar esta línea
    n_jobs=-1
)

print(f"\n   Configuración:")
print(f"   - Árboles: {modelo.n_estimators}")
print(f"   - Profundidad: {modelo.max_depth}")
print(f"   - Min split: {modelo.min_samples_split}")
print(f"   - Min leaf: {modelo.min_samples_leaf}")

modelo.fit(X_train, y_train)
print(f"\n   ✅ Entrenamiento completado")

# ============================================
# 8. EVALUAR MODELO
# ============================================
print("\n" + "="*80)
print("📊 EVALUACIÓN DEL MODELO")
print("="*80)

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n   Accuracy TRAIN: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Accuracy TEST:  {test_acc:.4f} ({test_acc*100:.2f}%)")

diff = train_acc - test_acc
if abs(diff) < 0.05:
    print(f"   ✅ Diferencia: {abs(diff):.4f} - Modelo generaliza bien")
elif abs(diff) < 0.15:
    print(f"   ⚠️  Diferencia: {abs(diff):.4f} - Ligero overfitting")
else:
    print(f"   ❌ Diferencia: {abs(diff):.4f} - Overfitting significativo")

# Métricas adicionales
precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"\n   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# Classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, zero_division=0))
report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
pd.DataFrame(report).transpose().to_csv('reportes/metricas_por_clase.csv')

# ============================================
# 9. CROSS-VALIDATION
# ============================================
print("\n🔄 Validación Cruzada (5-fold)...")

cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')

print(f"\n   Scores por fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"      Fold {i}: {score:.4f} ({score*100:.2f}%)")

print(f"\n   Media:  {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"   Std:    {cv_scores.std():.4f} (±{cv_scores.std()*100:.2f}%)")

# ============================================
# 10. FEATURE IMPORTANCE
# ============================================
print("\n" + "="*80)
print("🎯 IMPORTANCIA DE VARIABLES")
print("="*80)

importances = modelo.feature_importances_
feature_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n   Ranking:")
for idx, row in feature_imp.iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"      {row['feature']:20s} {row['importance']:.4f} {bar}")

# Guardar
feature_imp.to_csv('reportes/feature_importance.csv', index=False)
print(f"\n   ✅ Guardado: reportes/feature_importance.csv")

# ============================================
# 11. VISUALIZACIONES
# ============================================
print("\n" + "="*80)
print("📊 GENERANDO VISUALIZACIONES")
print("="*80)

# 1. Matriz de confusión
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

cm = confusion_matrix(y_test, y_pred_test, labels=['Bajo', 'Medio', 'Alto'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Bajo', 'Medio', 'Alto'],
            yticklabels=['Bajo', 'Medio', 'Alto'])
axes[0,0].set_title('Matriz de Confusión')
axes[0,0].set_ylabel('Real')
axes[0,0].set_xlabel('Predicción')

# 2. Feature importance
feature_imp_sorted = feature_imp.sort_values('importance', ascending=True)
axes[0,1].barh(feature_imp_sorted['feature'], feature_imp_sorted['importance'], color='skyblue')
axes[0,1].set_xlabel('Importancia')
axes[0,1].set_title('Importancia de Variables')

# 3. Distribución de clases
y_test.value_counts().plot(kind='bar', ax=axes[1,0], color='coral')
axes[1,0].set_title('Distribución Real (Test)')
axes[1,0].set_ylabel('Cantidad')
axes[1,0].set_xlabel('Nivel')

# 4. Accuracy por fold
axes[1,1].plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8)
axes[1,1].axhline(cv_scores.mean(), color='r', linestyle='--', label=f'Media: {cv_scores.mean():.3f}')
axes[1,1].set_xlabel('Fold')
axes[1,1].set_ylabel('Accuracy')
axes[1,1].set_title('Cross-Validation Scores')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos_ml/evaluacion_modelo.png', dpi=300, bbox_inches='tight')
print(f"   ✅ graficos_ml/evaluacion_modelo.png")
plt.close()

# ============================================
# 12. GUARDAR MODELO
# ============================================
print("\n" + "="*80)
print("💾 GUARDANDO MODELO Y ENCODERS")
print("="*80)

joblib.dump(modelo, 'modelos/modelo_ansiedad.pkl')
joblib.dump(le_sexo, 'modelos/encoder_sexo.pkl')
joblib.dump(le_estado_civil, 'modelos/encoder_estado_civil.pkl')
joblib.dump(le_carrera, 'modelos/encoder_carrera.pkl')

print(f"   ✅ Modelo: modelos/modelo_ansiedad.pkl")
print(f"   ✅ Encoders: modelos/encoder_*.pkl")

# Metadata
import json
metadata = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'n_registros': len(df),
    'n_features': len(feature_names),
    'features': feature_names,
    'clases': list(df['nivel_ansiedad'].unique()),
    'accuracy_test': float(test_acc),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'modelo_config': {
        'n_estimators': modelo.n_estimators,
        'max_depth': modelo.max_depth,
        'min_samples_split': modelo.min_samples_split,
        'min_samples_leaf': modelo.min_samples_leaf
    }
}

with open('modelos/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ Metadata: modelos/metadata.json")

# ============================================
# 13. RESUMEN FINAL
# ============================================
print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)

print(f"\n📊 RESUMEN:")
print(f"   Registros totales:       {len(df)}")
print(f"   Variables (features):    {len(feature_names)}")
print(f"   Clases únicas:           {clases_unicas}")
print(f"   ")
print(f"   Train set:               {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test set:                {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
print(f"   ")
print(f"   Accuracy (test):         {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   F1-Score:                {f1:.4f}")
print(f"   Cross-val (mean):        {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\n💡 INTERPRETACIÓN:")
if test_acc >= 0.90:
    if abs(train_acc - test_acc) > 0.10:
        print("   ⚠️  Accuracy muy alto + gran diferencia train/test = OVERFITTING")
        print("   → El modelo memorizó los datos en lugar de aprender")
        print("   → Reducir complejidad del modelo o aumentar datos")
    else:
        print("   ✅ Modelo con excelente rendimiento")
elif test_acc >= 0.70:
    print("   ✅ Modelo con buen rendimiento (accuracy aceptable)")
    print("   → Adecuado para sistema de screening de ansiedad")
else:
    print("   ⚠️  Accuracy bajo - revisar:")
    print("   → Calidad de los datos")
    print("   → Correlación entre variables y target")

print(f"\n🎯 PRÓXIMO PASO:")
print("   El modelo está listo para usar en tu API")
print("   Archivo: modelos/modelo_ansiedad.pkl")

print("\n" + "="*80)