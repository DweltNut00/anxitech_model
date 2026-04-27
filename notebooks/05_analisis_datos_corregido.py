import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ANÁLISIS DE DATASET - ANXITECH (VERSIÓN CORREGIDA)")
print("="*60)

# ============================================
# 1. CONEXIÓN
# ============================================
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="anxitech"
    )
    print("✅ Conexión exitosa\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ============================================
# 2. EXTRACCIÓN CON LEFT JOIN
# ============================================
print("📊 Extrayendo datos...\n")

# Query corregida: usar LEFT JOIN y tomar el último cuestionario
query = """
SELECT 
    c.id,
    c.id_alumno,
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
    ansiedad.promedio_ansiedad,
    ansiedad.suma_ansiedad,
    ansiedad.total_respuestas
FROM complemento c
LEFT JOIN (
    SELECT 
        id_alumno,
        AVG(valor) as promedio_ansiedad,
        SUM(valor) as suma_ansiedad,
        COUNT(*) as total_respuestas
    FROM alumno_pregunta
    GROUP BY id_alumno
) ansiedad ON c.id_alumno = ansiedad.id_alumno
ORDER BY c.id
"""

df = pd.read_sql(query, conn)

print(f"✅ Datos extraídos: {len(df)} registros\n")

# ============================================
# 3. INFORMACIÓN BÁSICA
# ============================================
print("="*60)
print("INFORMACIÓN GENERAL")
print("="*60)
print(f"Total de registros: {len(df)}")
print(f"Columnas: {len(df.columns)}")

# Verificar datos de ansiedad
registros_con_ansiedad = df['promedio_ansiedad'].notna().sum()
print(f"\n📊 Registros con datos de ansiedad: {registros_con_ansiedad}/{len(df)}")

if registros_con_ansiedad == 0:
    print("\n⚠️  NO HAY DATOS DE ANSIEDAD")
    print("Los alumnos con cuestionario complementario NO tienen respuestas del test DAS-42")
    print("\nVerifica:")
    conn.close()
    
    # Query de diagnóstico
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="anxitech")
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("DIAGNÓSTICO")
    print("="*60)
    
    cursor.execute("SELECT COUNT(DISTINCT id_alumno) FROM complemento")
    print(f"Alumnos con complemento: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(DISTINCT id_alumno) FROM alumno_pregunta")
    print(f"Alumnos con test ansiedad: {cursor.fetchone()[0]}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT c.id_alumno) 
        FROM complemento c 
        INNER JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
    """)
    print(f"Alumnos con AMBOS: {cursor.fetchone()[0]}")
    
    print("\n💡 SOLUCIÓN:")
    print("Necesitas que los mismos alumnos respondan AMBAS encuestas")
    
    cursor.close()
    conn.close()
    exit()

# ============================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# ============================================
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60)

numeric_cols = ['promedio_anterior', 'semestre', 'materias', 'edad', 'promedio_ansiedad']
print("\n📈 Variables Numéricas:")
print(df[numeric_cols].describe().round(2))

# ============================================
# 5. CLASIFICACIÓN DE ANSIEDAD
# ============================================
print("\n" + "="*60)
print("CLASIFICACIÓN DE NIVELES DE ANSIEDAD")
print("="*60)

# Clasificar en 3 niveles
df['nivel_ansiedad'] = pd.cut(
    df['promedio_ansiedad'], 
    bins=[0, 1.5, 2.5, 4.0], 
    labels=['Bajo', 'Medio', 'Alto'],
    include_lowest=True
)

print("\nDistribución de niveles:")
distribucion = df['nivel_ansiedad'].value_counts().sort_index()
for nivel, count in distribucion.items():
    porcentaje = (count / len(df)) * 100
    print(f"  {nivel}: {count} ({porcentaje:.1f}%)")

print(f"\nPromedio global: {df['promedio_ansiedad'].mean():.2f}")
print(f"Desviación estándar: {df['promedio_ansiedad'].std():.2f}")

# ============================================
# 6. ANÁLISIS POR CARRERA
# ============================================
print("\n" + "="*60)
print("ANSIEDAD POR CARRERA")
print("="*60)

ansiedad_carrera = df.groupby('carrera')['promedio_ansiedad'].agg(['mean', 'std', 'count'])
ansiedad_carrera.columns = ['Promedio', 'Desv.Std', 'N']
print("\n" + ansiedad_carrera.round(2).to_string())

# ============================================
# 7. CORRELACIONES
# ============================================
print("\n" + "="*60)
print("CORRELACIONES CON ANSIEDAD")
print("="*60)

correlation_cols = ['promedio_anterior', 'semestre', 'materias', 'edad', 
                    'transporte', 'trabajo', 'beca', 'promedio_ansiedad']

corr_matrix = df[correlation_cols].corr()

print("\nCorrelaciones con ansiedad:")
ansiedad_corr = corr_matrix['promedio_ansiedad'].drop('promedio_ansiedad').sort_values(ascending=False)
for var, corr in ansiedad_corr.items():
    if abs(corr) > 0.3:
        emoji = "🔴" if abs(corr) > 0.5 else "🟡"
        print(f"  {emoji} {var}: {corr:.3f}")
    else:
        print(f"  ⚪ {var}: {corr:.3f}")

# ============================================
# 8. VISUALIZACIONES
# ============================================
print("\n" + "="*60)
print("GENERANDO VISUALIZACIONES")
print("="*60)

import os
output_dir = 'analisis_dataset'
os.makedirs(output_dir, exist_ok=True)

# 1. Distribución de ansiedad
plt.figure(figsize=(10, 6))
plt.hist(df['promedio_ansiedad'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Promedio de Ansiedad', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de Niveles de Ansiedad', fontsize=14, fontweight='bold')
plt.axvline(df['promedio_ansiedad'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["promedio_ansiedad"].mean():.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/01_distribucion_ansiedad.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico 1: Distribución de ansiedad")

# 2. Ansiedad por carrera
if len(df['carrera'].unique()) > 1:
    plt.figure(figsize=(12, 6))
    df.boxplot(column='promedio_ansiedad', by='carrera', figsize=(12, 6))
    plt.xlabel('Carrera', fontsize=12)
    plt.ylabel('Promedio de Ansiedad', fontsize=12)
    plt.title('Nivel de Ansiedad por Carrera', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_ansiedad_por_carrera.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Ansiedad por carrera")

# 3. Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlaciones', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico 3: Matriz de correlación")

# 4. Niveles de ansiedad (pie chart)
plt.figure(figsize=(8, 8))
colors = ['#90EE90', '#FFD700', '#FF6B6B']
distribucion.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
plt.ylabel('')
plt.title('Distribución de Niveles de Ansiedad', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_niveles_ansiedad.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico 4: Niveles de ansiedad")

# ============================================
# 9. EXPORTAR DATASET
# ============================================
print("\n" + "="*60)
print("EXPORTANDO DATASET")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV
csv_file = f'{output_dir}/dataset_completo_{timestamp}.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"✅ CSV: {csv_file}")

# Excel con múltiples hojas
excel_file = f'{output_dir}/dataset_completo_{timestamp}.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Dataset', index=False)
    df[numeric_cols].describe().to_excel(writer, sheet_name='Estadísticas')
    ansiedad_carrera.to_excel(writer, sheet_name='Ansiedad por Carrera')
    corr_matrix.to_excel(writer, sheet_name='Correlaciones')

print(f"✅ Excel: {excel_file}")

# ============================================
# 10. REPORTE FINAL
# ============================================
print("\n" + "="*60)
print("REPORTE FINAL")
print("="*60)

print(f"""
✅ ANÁLISIS COMPLETADO

📊 Resumen:
- Registros totales: {len(df)}
- Con datos de ansiedad: {registros_con_ansiedad}
- Promedio de ansiedad: {df['promedio_ansiedad'].mean():.2f}
- Niveles: Bajo {distribucion.get('Bajo', 0)} | Medio {distribucion.get('Medio', 0)} | Alto {distribucion.get('Alto', 0)}

📁 Archivos generados:
- 4 gráficos PNG
- 1 archivo CSV
- 1 archivo Excel

💡 RECOMENDACIONES:
""")

if len(df) < 50:
    print("  ⚠️  Dataset pequeño (<50). Necesitas más datos reales.")
elif len(df) < 100:
    print("  ✅ Dataset adecuado para pruebas (~50-100).")
else:
    print("  ✅ Dataset robusto (>100 registros).")

if df['promedio_ansiedad'].std() < 0.5:
    print("  ⚠️  Poca variabilidad en ansiedad. Datos muy homogéneos.")
else:
    print("  ✅ Buena variabilidad en niveles de ansiedad.")

print("\n🚀 Listo para entrenar modelos de Machine Learning")
print("="*60)

conn.close()