import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ANÁLISIS DE DATASET - ANXITECH")
print("="*60)

# ============================================
# 1. CONEXIÓN A BASE DE DATOS
# ============================================
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Cambia según tu configuración
        password="",  # Cambia según tu configuración
        database="anxitech"
    )
    print("✅ Conexión exitosa a la base de datos\n")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    exit()

# ============================================
# 2. EXTRACCIÓN DE DATOS
# ============================================
print("📊 Extrayendo datos...")

query_complemento = """
SELECT 
    c.id,
    c.id_alumno,
    c.id_aplicacion,
    c.carrera,
    c.promedio_anterior,
    c.semestre,
    c.materias,
    c.transporte,
    c.familiares,
    c.trabajo,
    c.beca,
    c.sexo,
    c.edad,
    c.estado_civil,
    a.inicio as fecha_aplicacion,
    a.fin as fecha_fin
FROM complemento c
LEFT JOIN aplicacion a ON c.id_aplicacion = a.id
ORDER BY c.id
"""

query_ansiedad = """
SELECT 
    ap.id_alumno,
    ap.id_aplicacion,
    AVG(ap.valor) as promedio_ansiedad,
    SUM(ap.valor) as suma_ansiedad,
    COUNT(ap.id) as num_preguntas,
    p.categoria,
    AVG(CASE WHEN p.categoria = 'familia' THEN ap.valor END) as ansiedad_familia,
    AVG(CASE WHEN p.categoria = 'escuela' THEN ap.valor END) as ansiedad_escuela,
    AVG(CASE WHEN p.categoria = 'social' THEN ap.valor END) as ansiedad_social
FROM alumno_pregunta ap
LEFT JOIN pregunta p ON ap.id_pregunta = p.id
GROUP BY ap.id_alumno, ap.id_aplicacion
"""

# Extraer datos
df_complemento = pd.read_sql(query_complemento, conn)
df_ansiedad = pd.read_sql(query_ansiedad, conn)

# Unir datasets
df = pd.merge(
    df_complemento, 
    df_ansiedad, 
    on=['id_alumno', 'id_aplicacion'], 
    how='left'
)

print(f"✅ Datos extraídos: {len(df)} registros\n")

# ============================================
# 3. INFORMACIÓN BÁSICA
# ============================================
print("="*60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*60)
print(f"Total de registros: {len(df)}")
print(f"Total de columnas: {len(df.columns)}")
print(f"Rango de fechas: {df['fecha_aplicacion'].min()} a {df['fecha_aplicacion'].max()}")
print(f"\nColumnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================
# 4. ANÁLISIS DE VALORES FALTANTES
# ============================================
print("\n" + "="*60)
print("ANÁLISIS DE VALORES FALTANTES")
print("="*60)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Columna': missing.index,
    'Faltantes': missing.values,
    'Porcentaje': missing_percent.values
})
missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("✅ No hay valores faltantes")

# ============================================
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ============================================
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60)

# Variables numéricas
numeric_cols = ['promedio_anterior', 'semestre', 'materias', 'edad', 
                'promedio_ansiedad', 'suma_ansiedad']
print("\n📈 Variables Numéricas:")
print(df[numeric_cols].describe().round(2))

# Variables categóricas
print("\n📋 Variables Categóricas:")
categorical_cols = ['carrera', 'sexo', 'estado_civil', 'transporte', 
                    'familiares', 'trabajo', 'beca']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col.upper()}:")
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            percent = (count / len(df)) * 100
            print(f"  {val}: {count} ({percent:.1f}%)")

# ============================================
# 6. ANÁLISIS DE ANSIEDAD
# ============================================
if 'promedio_ansiedad' in df.columns and df['promedio_ansiedad'].notna().sum() > 0:
    print("\n" + "="*60)
    print("ANÁLISIS DE NIVELES DE ANSIEDAD")
    print("="*60)
    
    # Clasificación de ansiedad (ajusta según tu escala)
    df['nivel_ansiedad'] = pd.cut(
        df['promedio_ansiedad'], 
        bins=[0, 1, 2, 3, 4], 
        labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
    )
    
    print("\nDistribución de niveles de ansiedad:")
    print(df['nivel_ansiedad'].value_counts().sort_index())
    
    print(f"\nPromedio general de ansiedad: {df['promedio_ansiedad'].mean():.2f}")
    print(f"Desviación estándar: {df['promedio_ansiedad'].std():.2f}")
    print(f"Mínimo: {df['promedio_ansiedad'].min():.2f}")
    print(f"Máximo: {df['promedio_ansiedad'].max():.2f}")

# ============================================
# 7. CORRELACIONES
# ============================================
print("\n" + "="*60)
print("MATRIZ DE CORRELACIONES")
print("="*60)

correlation_cols = ['promedio_anterior', 'semestre', 'materias', 'edad', 
                    'transporte', 'trabajo', 'beca', 'promedio_ansiedad']
correlation_cols = [col for col in correlation_cols if col in df.columns]

if len(correlation_cols) > 0:
    corr_matrix = df[correlation_cols].corr()
    print("\nCorrelaciones con 'promedio_ansiedad':")
    if 'promedio_ansiedad' in corr_matrix.columns:
        ansiedad_corr = corr_matrix['promedio_ansiedad'].sort_values(ascending=False)
        print(ansiedad_corr.to_string())

# ============================================
# 8. DETECCIÓN DE OUTLIERS
# ============================================
print("\n" + "="*60)
print("DETECCIÓN DE OUTLIERS")
print("="*60)

for col in ['promedio_anterior', 'edad', 'promedio_ansiedad']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\n{col}:")
        print(f"  Rango normal: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# ============================================
# 9. VALIDACIÓN DE DATOS SINTÉTICOS
# ============================================
print("\n" + "="*60)
print("VALIDACIÓN DE CALIDAD DE DATOS")
print("="*60)

issues = []

# Validar rangos de variables
if 'promedio_anterior' in df.columns:
    invalid_promedios = df[(df['promedio_anterior'] < 0) | (df['promedio_anterior'] > 100)]
    if len(invalid_promedios) > 0:
        issues.append(f"❌ {len(invalid_promedios)} promedios fuera de rango [0-100]")
    else:
        print("✅ Promedios en rango válido")

if 'semestre' in df.columns:
    invalid_semestres = df[(df['semestre'] < 1) | (df['semestre'] > 13)]
    if len(invalid_semestres) > 0:
        issues.append(f"❌ {len(invalid_semestres)} semestres fuera de rango [1-13]")
    else:
        print("✅ Semestres en rango válido")

if 'edad' in df.columns:
    invalid_edades = df[(df['edad'] < 17) | (df['edad'] > 50)]
    if len(invalid_edades) > 0:
        issues.append(f"⚠️  {len(invalid_edades)} edades atípicas para universitarios")
    else:
        print("✅ Edades en rango típico")

if 'materias' in df.columns:
    invalid_materias = df[(df['materias'] < 1) | (df['materias'] > 9)]
    if len(invalid_materias) > 0:
        issues.append(f"⚠️  {len(invalid_materias)} cantidades de materias atípicas")
    else:
        print("✅ Cantidad de materias en rango típico")

# Verificar distribuciones realistas
if 'sexo' in df.columns:
    sexo_dist = df['sexo'].value_counts(normalize=True)
    if len(sexo_dist) == 2 and 0.3 < sexo_dist.min() < 0.7:
        print("✅ Distribución de sexo balanceada")
    else:
        issues.append("⚠️  Distribución de sexo muy desbalanceada")

if len(issues) > 0:
    print("\n⚠️  PROBLEMAS DETECTADOS:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ Todos los datos pasan las validaciones básicas")

# ============================================
# 10. VISUALIZACIONES
# ============================================
print("\n" + "="*60)
print("GENERANDO VISUALIZACIONES")
print("="*60)

# Crear carpeta para guardar gráficos
import os
output_dir = 'analisis_dataset'
os.makedirs(output_dir, exist_ok=True)

# 1. Distribución de ansiedad
if 'promedio_ansiedad' in df.columns and df['promedio_ansiedad'].notna().sum() > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(df['promedio_ansiedad'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Promedio de Ansiedad')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Niveles de Ansiedad')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/01_distribucion_ansiedad.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1: Distribución de ansiedad")

# 2. Ansiedad por carrera
if 'promedio_ansiedad' in df.columns and 'carrera' in df.columns:
    plt.figure(figsize=(12, 6))
    df.boxplot(column='promedio_ansiedad', by='carrera', figsize=(12, 6))
    plt.xlabel('Carrera')
    plt.ylabel('Promedio de Ansiedad')
    plt.title('Nivel de Ansiedad por Carrera')
    plt.suptitle('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_ansiedad_por_carrera.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Ansiedad por carrera")

# 3. Matriz de correlación
if len(correlation_cols) > 0:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_matriz_correlacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3: Matriz de correlación")

# 4. Distribución de variables categóricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
categorical_plots = ['sexo', 'estado_civil', 'trabajo', 'beca']

for idx, col in enumerate(categorical_plots):
    if col in df.columns:
        ax = axes[idx // 2, idx % 2]
        df[col].value_counts().plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.set_ylabel('Frecuencia')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_variables_categoricas.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico 4: Variables categóricas")

# ============================================
# 11. EXPORTAR DATASET
# ============================================
print("\n" + "="*60)
print("EXPORTANDO DATASET")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV
csv_filename = f'{output_dir}/dataset_anxitech_{timestamp}.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"✅ CSV guardado: {csv_filename}")

# Excel
excel_filename = f'{output_dir}/dataset_anxitech_{timestamp}.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Dataset Completo', index=False)
    df[numeric_cols].describe().to_excel(writer, sheet_name='Estadísticas')
    if 'promedio_ansiedad' in df.columns:
        df.groupby('carrera')['promedio_ansiedad'].agg(['mean', 'std', 'min', 'max']).to_excel(
            writer, sheet_name='Ansiedad por Carrera'
        )

print(f"✅ Excel guardado: {excel_filename}")

# ============================================
# 12. REPORTE FINAL
# ============================================
print("\n" + "="*60)
print("REPORTE FINAL")
print("="*60)

print(f"""
✅ ANÁLISIS COMPLETADO

📊 Resumen:
- Total de registros: {len(df)}
- Variables numéricas: {len([c for c in numeric_cols if c in df.columns])}
- Variables categóricas: {len([c for c in categorical_cols if c in df.columns])}
- Registros con ansiedad: {df['promedio_ansiedad'].notna().sum() if 'promedio_ansiedad' in df.columns else 0}

📁 Archivos generados en '{output_dir}/':
- 4 gráficos en formato PNG
- 1 archivo CSV
- 1 archivo Excel con múltiples hojas

💡 RECOMENDACIONES:
""")

if len(df) < 100:
    print("  ⚠️  Dataset pequeño. Considera generar más datos sintéticos o esperar datos reales.")
elif len(df) < 500:
    print("  ✅ Dataset adecuado para pruebas iniciales.")
else:
    print("  ✅ Dataset robusto para entrenamiento de modelos.")

if len(issues) > 0:
    print("  ⚠️  Revisar problemas de calidad detectados arriba.")
else:
    print("  ✅ Calidad de datos validada exitosamente.")

if 'promedio_ansiedad' in df.columns and df['promedio_ansiedad'].notna().sum() > 50:
    print("  ✅ Suficientes datos de ansiedad para entrenar modelo.")
else:
    print("  ⚠️  Pocos datos de ansiedad. Necesitas más encuestas respondidas.")

print("\n🚀 Siguiente paso: Entrenar modelos de machine learning")
print("="*60)

conn.close()