"""
ENTRENAMIENTO DE MODELO ML - ANXITECH (VERSIÓN DEFINITIVA)
============================================================
Con umbrales CORREGIDOS para escala 0-10

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
print("ENTRENAMIENTO DE MODELO - RANDOM FOREST")
print("="*80)

# ============================================
# FUNCIÓN DE CLASIFICACIÓN CORREGIDA
# ============================================
def clasificar_ansiedad(suma):
    """
    Clasifica nivel de ansiedad según suma DASS-21 (7 preguntas)
    
    IMPORTANTE: Ajustado para escala 0-10 (rango observado en poblaciones reales)
    - Teórico: 7 preguntas × 3 puntos = 0-21
    - Real: En poblaciones no clínicas raramente se alcanza 21
    - Observado: 0-10 puntos
    
    Umbrales ajustados proporcionalmente:
    - Bajo:  0-4  (equivalente a 0-9 en escala completa)
    - Medio: 5-7  (equivalente a 10-14 en escala completa)
    - Alto:  8-10 (equivalente a 15-21 en escala completa)
    
    Args:
        suma (int/float): Suma de 7 respuestas DASS-21 (0-10)
    
    Returns:
        str: 'Bajo', 'Medio' o 'Alto'
    """
    if suma <= 4:
        return 'Bajo'
    elif suma <= 7:
        return 'Medio'
    else:  # 8-10
        return 'Alto'

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
        exit()
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ============================================
# 3. CLASIFICAR CON UMBRALES CORRECTOS
# ============================================
print("\n📊 Clasificando niveles de ansiedad (umbrales 0-10)...")

print(f"\n   ⚙️  Umbrales configurados:")
print(f"   Bajo:  suma ≤ 4  (0-4 puntos)")
print(f"   Medio: suma 5-7  (5-7 puntos)")
print(f"   Alto:  suma ≥ 8  (8-10 puntos)")

df['nivel_ansiedad'] = df['suma_ansiedad'].apply(clasificar_ansiedad)

print(f"\n   📊 Rango de sumas:")
print(f"   Mínimo:  {df['suma_ansiedad'].min():.0f}")
print(f"   Máximo:  {df['suma_ansiedad'].max():.0f}")
print(f"   Media:   {df['suma_ansiedad'].mean():.2f}")
print(f"   Mediana: {df['suma_ansiedad'].median():.2f}")

print(f"\n   📊 Distribución de niveles:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = (df['nivel_ansiedad'] == nivel).sum()
    pct = (count / len(df)) * 100
    emoji = {'Bajo': '🟢', 'Medio': '🟡', 'Alto': '🔴'}[nivel]
    print(f"      {emoji} {nivel}: {count} ({pct:.1f}%)")

# Verificar clases
clases_unicas = df['nivel_ansiedad'].nunique()
if clases_unicas < 2:
    print(f"\n❌ ERROR: Solo hay {clases_unicas} clase(s)")
    exit()

print(f"\n   ✅ {clases_unicas} clases detectadas")

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
# 5. CODIFICAR VARIABLES
# ============================================
print("\n🔤 Codificando variables categóricas...")

le_sexo = LabelEncoder()
le_estado_civil = LabelEncoder()
le_carrera = LabelEncoder()

X['sexo'] = le_sexo.fit_transform(X['sexo'].astype(str))
X['estado_civil'] = le_estado_civil.fit_transform(X['estado_civil'].astype(str))
X['carrera'] = le_carrera.fit_transform(X['carrera'].astype(str))

print(f"\n   ✅ Variables codificadas")

# Verificar valores nulos
if X.isnull().sum().sum() > 0:
    print(f"\n⚠️  Imputando valores nulos...")
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
print("\n📊 Dividiendo datos (80/20)...")

# Verificar muestras mínimas por clase
min_samples = y.value_counts().min()
print(f"   Clase con menos muestras: {min_samples}")

if min_samples < 10:
    print(f"   ⚠️  Pocas muestras en alguna clase, usando split sin stratify")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"\n   ✅ Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ✅ Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Mostrar distribución en train/test
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
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
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
# 8. EVALUAR
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

precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"\n   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, zero_division=0))

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

feature_imp.to_csv('reportes/feature_importance.csv', index=False)
print(f"\n   ✅ Guardado: reportes/feature_importance.csv")

# ============================================
# 11. GUARDAR MODELO
# ============================================
print("\n💾 Guardando modelo...")

joblib.dump(modelo, 'modelos/modelo_ansiedad.pkl')
joblib.dump(le_sexo, 'modelos/encoder_sexo.pkl')
joblib.dump(le_estado_civil, 'modelos/encoder_estado_civil.pkl')
joblib.dump(le_carrera, 'modelos/encoder_carrera.pkl')

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
    'umbrales_clasificacion': {
        'Bajo': '≤ 4 puntos',
        'Medio': '5-7 puntos',
        'Alto': '≥ 8 puntos'
    },
    'escala': 'DASS-21 (7 preguntas de ansiedad, rango 0-10)',
    'modelo_config': {
        'n_estimators': modelo.n_estimators,
        'max_depth': modelo.max_depth,
        'min_samples_split': modelo.min_samples_split,
        'min_samples_leaf': modelo.min_samples_leaf
    }
}

with open('modelos/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ Modelo y metadata guardados")

# ============================================
# 12. VISUALIZACIÓN
# ============================================
print("\n📊 Generando visualización...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test, labels=['Bajo', 'Medio', 'Alto'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Bajo', 'Medio', 'Alto'],
            yticklabels=['Bajo', 'Medio', 'Alto'])
axes[0,0].set_title('Matriz de Confusión')
axes[0,0].set_ylabel('Real')
axes[0,0].set_xlabel('Predicción')

# Feature importance
feature_imp_sorted = feature_imp.sort_values('importance', ascending=True)
axes[0,1].barh(feature_imp_sorted['feature'], feature_imp_sorted['importance'], color='skyblue')
axes[0,1].set_xlabel('Importancia')
axes[0,1].set_title('Importancia de Variables')

# Distribución
y_test.value_counts().plot(kind='bar', ax=axes[1,0], color='coral')
axes[1,0].set_title('Distribución Real (Test)')
axes[1,0].set_ylabel('Cantidad')
axes[1,0].set_xlabel('Nivel')

# CV Scores
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
# 13. RESUMEN
# ============================================
print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)

print(f"\n📊 RESUMEN:")
print(f"   Registros totales:       {len(df)}")
print(f"   Variables:               {len(feature_names)}")
print(f"   Clases:                  {clases_unicas}")
print(f"   ")
print(f"   Train set:               {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test set:                {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
print(f"   ")
print(f"   Accuracy (test):         {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   F1-Score:                {f1:.4f}")
print(f"   Cross-val (mean):        {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\n💡 INTERPRETACIÓN:")
if test_acc >= 0.90 and clases_unicas == 1:
    print("   ❌ Accuracy alto pero 1 sola clase = problema de datos")
elif test_acc >= 0.90 and abs(train_acc - test_acc) > 0.10:
    print("   ⚠️  Posible overfitting - revisar complejidad del modelo")
elif test_acc >= 0.70:
    print("   ✅ Modelo con buen rendimiento para screening de ansiedad")
    print("   → Adecuado para identificar estudiantes que necesitan atención")
else:
    print("   ⚠️  Accuracy moderado - revisar calidad de datos y features")

print(f"\n🎯 MODELO LISTO:")
print("   Archivo: modelos/modelo_ansiedad.pkl")
print("   Úsalo en tu API para predicciones en tiempo real")

print("\n" + "="*80)