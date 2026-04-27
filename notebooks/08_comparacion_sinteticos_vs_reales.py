"""
COMPARACIÓN: DATOS SINTÉTICOS vs DATOS REALES
==============================================
Compara 500 estudiantes sintéticos (BD AnxiTech) con 570 estudiantes 
universitarios reales (CSV DASS-21 Bangladesh)

Autor: Sistema AnxiTech
Fecha: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mysql.connector
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMPARACIÓN: DATOS SINTÉTICOS vs DATOS REALES")
print("="*80)

# ============================================
# 1. CARGAR DATOS SINTÉTICOS (BASE DE DATOS)
# ============================================
print("\n📊 CARGANDO DATOS SINTÉTICOS (BD AnxiTech)...")

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="anxitech"
    )
    
    # Query para obtener datos sintéticos con respuestas DASS-21
    query_sinteticos = """
    SELECT 
        c.id_alumno,
        c.carrera,
        c.promedio_anterior,
        c.semestre,
        c.materias,
        c.edad,
        c.sexo,
        AVG(ap.valor) as promedio_respuesta,
        SUM(ap.valor) as suma_ansiedad,
        COUNT(ap.id) as num_respuestas
    FROM complemento c
    INNER JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
    INNER JOIN pregunta p ON ap.id_pregunta = p.id
    WHERE p.categoria = 'ansiedad' AND p.status = 1
    GROUP BY c.id_alumno
    HAVING COUNT(ap.id) = 7
    ORDER BY c.id_alumno
    """
    
    df_sinteticos = pd.read_sql(query_sinteticos, conn)
    conn.close()
    
    print(f"✅ Datos sintéticos cargados: {len(df_sinteticos)} estudiantes")
    
    if len(df_sinteticos) == 0:
        print("\n❌ ERROR: No hay datos sintéticos en la BD")
        print("   Ejecuta primero: python 04_generar_respuestas_sinteticas.py")
        exit()
        
except Exception as e:
    print(f"❌ Error conectando a BD: {e}")
    exit()

# ============================================
# 2. CARGAR DATOS REALES (CSV)
# ============================================
print("\n📊 CARGANDO DATOS REALES (CSV DASS-21)...")

try:
    # Ajusta la ruta si es necesario
    df_reales_completo = pd.read_csv('DASS.csv')
    
    # Filtrar solo estudiantes universitarios (Q1_4 == 5)
    df_reales = df_reales_completo[df_reales_completo['Q1_4'] == 5].copy()
    
    print(f"✅ Dataset completo: {len(df_reales_completo)} registros")
    print(f"✅ Estudiantes universitarios: {len(df_reales)} registros")
    
    # Renombrar columnas para facilitar análisis
    df_reales['edad'] = df_reales['Q1_1']
    df_reales['sexo'] = df_reales['Q1_2'].map({1: 'M', 2: 'F'})
    df_reales['suma_ansiedad'] = df_reales['Anxiety_Score']
    df_reales['nivel_original'] = df_reales['Anxiety_Level']
    
    # Calcular promedio de respuestas
    columnas_ansiedad = ['Q3_8_A1', 'Q3_9_A2', 'Q3_10_A3', 'Q3_11_A4', 
                         'Q3_12_A5', 'Q3_13_A6', 'Q3_14_A7']
    df_reales['promedio_respuesta'] = df_reales[columnas_ansiedad].mean(axis=1)
    
except FileNotFoundError:
    print("❌ ERROR: Archivo DASS.csv no encontrado")
    print("   Coloca el archivo en la misma carpeta que este script")
    exit()
except Exception as e:
    print(f"❌ Error cargando CSV: {e}")
    exit()

# ============================================
# 3. RECLASIFICAR NIVELES (5 NIVELES → 3 NIVELES)
# ============================================
print("\n" + "="*80)
print("RECLASIFICACIÓN DE NIVELES: 5 NIVELES → 3 NIVELES")
print("="*80)

def reclasificar_5_a_3_niveles(nivel_5):
    """
    Convierte niveles DASS-21 de 5 categorías a 3 categorías
    
    DASS-21 original (5 niveles):
    1 = Normal (0-7)
    2 = Leve (8-9)
    3 = Moderado (10-14)
    4 = Severo (15-19)
    5 = Extremadamente severo (20-21)
    
    Sistema AnxiTech (3 niveles):
    Bajo = Normal + Leve (0-9)
    Medio = Moderado (10-14)
    Alto = Severo + Extremo (15-21)
    """
    if nivel_5 in [1, 2]:  # Normal o Leve
        return 'Bajo'
    elif nivel_5 == 3:      # Moderado
        return 'Medio'
    else:                   # Severo o Extremo (4, 5)
        return 'Alto'

def clasificar_por_suma(suma):
    """
    Clasifica por suma total (0-21 puntos)
    Basado en rangos DASS-21 estándar
    """
    if suma <= 9:
        return 'Bajo'
    elif suma <= 14:
        return 'Medio'
    else:
        return 'Alto'

# Aplicar reclasificación a datos reales
df_reales['nivel_3'] = df_reales['nivel_original'].apply(reclasificar_5_a_3_niveles)

# Clasificar datos sintéticos
df_sinteticos['nivel_3'] = df_sinteticos['suma_ansiedad'].apply(clasificar_por_suma)

print("\n📊 MAPEO DE 5 A 3 NIVELES:")
print("   Nivel 1 (Normal)          → Bajo")
print("   Nivel 2 (Leve)            → Bajo")
print("   Nivel 3 (Moderado)        → Medio")
print("   Nivel 4 (Severo)          → Alto")
print("   Nivel 5 (Extremo)         → Alto")

print("\n📊 DISTRIBUCIÓN DATOS REALES:")
print("\nOriginal (5 niveles):")
print(df_reales['nivel_original'].value_counts().sort_index())

print("\nReclasificado (3 niveles):")
dist_reales = df_reales['nivel_3'].value_counts()
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = dist_reales.get(nivel, 0)
    pct = (count / len(df_reales)) * 100
    print(f"   {nivel:<8}: {count:3d} ({pct:5.1f}%)")

print("\n📊 DISTRIBUCIÓN DATOS SINTÉTICOS (3 niveles):")
dist_sinteticos = df_sinteticos['nivel_3'].value_counts()
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = dist_sinteticos.get(nivel, 0)
    pct = (count / len(df_sinteticos)) * 100
    print(f"   {nivel:<8}: {count:3d} ({pct:5.1f}%)")

# ============================================
# 4. ESTADÍSTICAS COMPARATIVAS
# ============================================
print("\n" + "="*80)
print("ESTADÍSTICAS COMPARATIVAS")
print("="*80)

# Crear DataFrame comparativo
comparacion = pd.DataFrame({
    'Métrica': [
        'N (muestra)',
        'Suma media',
        'Suma desv.std',
        'Suma mínimo',
        'Suma máximo',
        'Promedio/pregunta',
        '% Bajo',
        '% Medio',
        '% Alto'
    ],
    'Sintéticos (n=500)': [
        len(df_sinteticos),
        f"{df_sinteticos['suma_ansiedad'].mean():.2f}",
        f"{df_sinteticos['suma_ansiedad'].std():.2f}",
        f"{df_sinteticos['suma_ansiedad'].min():.0f}",
        f"{df_sinteticos['suma_ansiedad'].max():.0f}",
        f"{df_sinteticos['promedio_respuesta'].mean():.2f}",
        f"{(dist_sinteticos.get('Bajo', 0) / len(df_sinteticos) * 100):.1f}%",
        f"{(dist_sinteticos.get('Medio', 0) / len(df_sinteticos) * 100):.1f}%",
        f"{(dist_sinteticos.get('Alto', 0) / len(df_sinteticos) * 100):.1f}%"
    ],
    'Reales (n=570)': [
        len(df_reales),
        f"{df_reales['suma_ansiedad'].mean():.2f}",
        f"{df_reales['suma_ansiedad'].std():.2f}",
        f"{df_reales['suma_ansiedad'].min():.0f}",
        f"{df_reales['suma_ansiedad'].max():.0f}",
        f"{df_reales['promedio_respuesta'].mean():.2f}",
        f"{(dist_reales.get('Bajo', 0) / len(df_reales) * 100):.1f}%",
        f"{(dist_reales.get('Medio', 0) / len(df_reales) * 100):.1f}%",
        f"{(dist_reales.get('Alto', 0) / len(df_reales) * 100):.1f}%"
    ]
})

print("\n" + comparacion.to_string(index=False))

# ============================================
# 5. PRUEBAS ESTADÍSTICAS
# ============================================
print("\n" + "="*80)
print("PRUEBAS ESTADÍSTICAS")
print("="*80)

# Test de Kolmogorov-Smirnov (comparar distribuciones)
ks_statistic, ks_pvalue = stats.ks_2samp(
    df_sinteticos['suma_ansiedad'], 
    df_reales['suma_ansiedad']
)

print("\n📊 Test Kolmogorov-Smirnov (distribuciones):")
print(f"   Estadístico: {ks_statistic:.4f}")
print(f"   p-valor: {ks_pvalue:.4f}")
if ks_pvalue < 0.05:
    print(f"   ⚠️  Las distribuciones son SIGNIFICATIVAMENTE DIFERENTES (p < 0.05)")
else:
    print(f"   ✅ Las distribuciones NO son significativamente diferentes (p >= 0.05)")

# Test t de Student (comparar medias)
t_statistic, t_pvalue = stats.ttest_ind(
    df_sinteticos['suma_ansiedad'],
    df_reales['suma_ansiedad']
)

print("\n📊 Test t de Student (medias):")
print(f"   Estadístico t: {t_statistic:.4f}")
print(f"   p-valor: {t_pvalue:.4f}")
if t_pvalue < 0.05:
    print(f"   ⚠️  Las medias son SIGNIFICATIVAMENTE DIFERENTES (p < 0.05)")
else:
    print(f"   ✅ Las medias NO son significativamente diferentes (p >= 0.05)")

# Chi-cuadrado (comparar distribución de niveles)
# Crear tabla de contingencia
contingencia = pd.DataFrame({
    'Sintéticos': [
        dist_sinteticos.get('Bajo', 0),
        dist_sinteticos.get('Medio', 0),
        dist_sinteticos.get('Alto', 0)
    ],
    'Reales': [
        dist_reales.get('Bajo', 0),
        dist_reales.get('Medio', 0),
        dist_reales.get('Alto', 0)
    ]
}, index=['Bajo', 'Medio', 'Alto'])

chi2, chi_pvalue, dof, expected = stats.chi2_contingency(contingencia.T)

print("\n📊 Test Chi-cuadrado (proporciones de niveles):")
print(f"   Chi-cuadrado: {chi2:.4f}")
print(f"   p-valor: {chi_pvalue:.4f}")
print(f"   Grados de libertad: {dof}")
if chi_pvalue < 0.05:
    print(f"   ⚠️  Las proporciones son SIGNIFICATIVAMENTE DIFERENTES (p < 0.05)")
else:
    print(f"   ✅ Las proporciones NO son significativamente diferentes (p >= 0.05)")

# ============================================
# 6. VISUALIZACIONES COMPARATIVAS
# ============================================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

import os
output_dir = 'comparacion_sinteticos_reales'
os.makedirs(output_dir, exist_ok=True)

# Figura con 6 subgráficos
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Histogramas comparativos de suma de ansiedad
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df_sinteticos['suma_ansiedad'], bins=22, alpha=0.6, label='Sintéticos (n=500)', 
         color='steelblue', edgecolor='black')
ax1.hist(df_reales['suma_ansiedad'], bins=22, alpha=0.6, label='Reales (n=570)', 
         color='coral', edgecolor='black')
ax1.axvline(df_sinteticos['suma_ansiedad'].mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Media Sint: {df_sinteticos["suma_ansiedad"].mean():.1f}')
ax1.axvline(df_reales['suma_ansiedad'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media Real: {df_reales["suma_ansiedad"].mean():.1f}')
ax1.set_xlabel('Suma de Ansiedad (0-21)', fontsize=11)
ax1.set_ylabel('Frecuencia', fontsize=11)
ax1.set_title('Distribución de Puntajes DASS-21', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Boxplot comparativo
ax2 = fig.add_subplot(gs[0, 2])
data_box = [df_sinteticos['suma_ansiedad'], df_reales['suma_ansiedad']]
bp = ax2.boxplot(data_box, labels=['Sintéticos', 'Reales'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax2.set_ylabel('Suma de Ansiedad', fontsize=11)
ax2.set_title('Comparación de Distribuciones', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Distribución de niveles - Sintéticos
ax3 = fig.add_subplot(gs[1, 0])
niveles_orden = ['Bajo', 'Medio', 'Alto']
colores_niveles = ['#90EE90', '#FFD700', '#FF6B6B']
counts_sint = [dist_sinteticos.get(n, 0) for n in niveles_orden]
ax3.bar(niveles_orden, counts_sint, color=colores_niveles, edgecolor='black', alpha=0.8)
for i, (nivel, count) in enumerate(zip(niveles_orden, counts_sint)):
    pct = (count / len(df_sinteticos)) * 100
    ax3.text(i, count + 5, f'{count}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frecuencia', fontsize=11)
ax3.set_title('Sintéticos (n=500)', fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(counts_sint) * 1.15)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Distribución de niveles - Reales
ax4 = fig.add_subplot(gs[1, 1])
counts_real = [dist_reales.get(n, 0) for n in niveles_orden]
ax4.bar(niveles_orden, counts_real, color=colores_niveles, edgecolor='black', alpha=0.8)
for i, (nivel, count) in enumerate(zip(niveles_orden, counts_real)):
    pct = (count / len(df_reales)) * 100
    ax4.text(i, count + 5, f'{count}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax4.set_ylabel('Frecuencia', fontsize=11)
ax4.set_title('Reales (n=570)', fontsize=13, fontweight='bold')
ax4.set_ylim(0, max(counts_real) * 1.15)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Comparación porcentual
ax5 = fig.add_subplot(gs[1, 2])
x = np.arange(len(niveles_orden))
width = 0.35
pct_sint = [(dist_sinteticos.get(n, 0) / len(df_sinteticos)) * 100 for n in niveles_orden]
pct_real = [(dist_reales.get(n, 0) / len(df_reales)) * 100 for n in niveles_orden]
ax5.bar(x - width/2, pct_sint, width, label='Sintéticos', color='steelblue', edgecolor='black')
ax5.bar(x + width/2, pct_real, width, label='Reales', color='coral', edgecolor='black')
ax5.set_ylabel('Porcentaje (%)', fontsize=11)
ax5.set_title('Comparación Porcentual', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(niveles_orden)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Q-Q Plot
ax6 = fig.add_subplot(gs[2, 0])
stats.probplot(df_sinteticos['suma_ansiedad'], dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot: Sintéticos', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Q-Q Plot Reales
ax7 = fig.add_subplot(gs[2, 1])
stats.probplot(df_reales['suma_ansiedad'], dist="norm", plot=ax7)
ax7.set_title('Q-Q Plot: Reales', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Tabla de comparación
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('tight')
ax8.axis('off')
tabla_data = [
    ['Métrica', 'Sintéticos', 'Reales'],
    ['n', f'{len(df_sinteticos)}', f'{len(df_reales)}'],
    ['Media', f'{df_sinteticos["suma_ansiedad"].mean():.2f}', f'{df_reales["suma_ansiedad"].mean():.2f}'],
    ['Desv.Std', f'{df_sinteticos["suma_ansiedad"].std():.2f}', f'{df_reales["suma_ansiedad"].std():.2f}'],
    ['Min-Max', f'{df_sinteticos["suma_ansiedad"].min():.0f}-{df_sinteticos["suma_ansiedad"].max():.0f}', 
                f'{df_reales["suma_ansiedad"].min():.0f}-{df_reales["suma_ansiedad"].max():.0f}'],
    ['% Bajo', f'{pct_sint[0]:.1f}%', f'{pct_real[0]:.1f}%'],
    ['% Medio', f'{pct_sint[1]:.1f}%', f'{pct_real[1]:.1f}%'],
    ['% Alto', f'{pct_sint[2]:.1f}%', f'{pct_real[2]:.1f}%']
]
table = ax8.table(cellText=tabla_data, cellLoc='center', loc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Header styling
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.suptitle('COMPARACIÓN: DATOS SINTÉTICOS vs DATOS REALES\nDASS-21 Ansiedad', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(f'{output_dir}/comparacion_completa.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {output_dir}/comparacion_completa.png")
plt.close()

# ============================================
# 7. EXPORTAR RESULTADOS
# ============================================
print("\n" + "="*80)
print("EXPORTANDO RESULTADOS")
print("="*80)

# Guardar datasets procesados
df_sinteticos[['id_alumno', 'suma_ansiedad', 'promedio_respuesta', 'nivel_3']].to_csv(
    f'{output_dir}/sinteticos_procesados.csv', index=False
)
print(f"✅ {output_dir}/sinteticos_procesados.csv")

df_reales[['edad', 'sexo', 'suma_ansiedad', 'promedio_respuesta', 'nivel_original', 'nivel_3']].to_csv(
    f'{output_dir}/reales_procesados.csv', index=False
)
print(f"✅ {output_dir}/reales_procesados.csv")

# Guardar tabla comparativa
comparacion.to_csv(f'{output_dir}/tabla_comparativa.csv', index=False)
print(f"✅ {output_dir}/tabla_comparativa.csv")

# Guardar resultados estadísticos
with open(f'{output_dir}/pruebas_estadisticas.txt', 'w', encoding='utf-8') as f:
    f.write("PRUEBAS ESTADÍSTICAS - COMPARACIÓN SINTÉTICOS vs REALES\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Kolmogorov-Smirnov:\n")
    f.write(f"  Estadístico: {ks_statistic:.4f}\n")
    f.write(f"  p-valor: {ks_pvalue:.4f}\n")
    f.write(f"  Interpretación: {'Distribuciones diferentes' if ks_pvalue < 0.05 else 'Distribuciones similares'}\n\n")
    
    f.write(f"Test t de Student:\n")
    f.write(f"  Estadístico t: {t_statistic:.4f}\n")
    f.write(f"  p-valor: {t_pvalue:.4f}\n")
    f.write(f"  Interpretación: {'Medias diferentes' if t_pvalue < 0.05 else 'Medias similares'}\n\n")
    
    f.write(f"Test Chi-cuadrado:\n")
    f.write(f"  Chi-cuadrado: {chi2:.4f}\n")
    f.write(f"  p-valor: {chi_pvalue:.4f}\n")
    f.write(f"  Interpretación: {'Proporciones diferentes' if chi_pvalue < 0.05 else 'Proporciones similares'}\n")

print(f"✅ {output_dir}/pruebas_estadisticas.txt")

# ============================================
# 8. REPORTE FINAL
# ============================================
print("\n" + "="*80)
print("REPORTE FINAL DE COMPARACIÓN")
print("="*80)

diferencia_media = abs(df_sinteticos['suma_ansiedad'].mean() - df_reales['suma_ansiedad'].mean())
diferencia_pct_bajo = abs(pct_sint[0] - pct_real[0])
diferencia_pct_medio = abs(pct_sint[1] - pct_real[1])
diferencia_pct_alto = abs(pct_sint[2] - pct_real[2])

print(f"""
📊 RESUMEN DE COMPARACIÓN

TAMAÑOS DE MUESTRA:
- Datos sintéticos: {len(df_sinteticos)} estudiantes
- Datos reales:     {len(df_reales)} estudiantes universitarios

PUNTAJES DE ANSIEDAD (0-21):
- Media sintéticos: {df_sinteticos['suma_ansiedad'].mean():.2f} ± {df_sinteticos['suma_ansiedad'].std():.2f}
- Media reales:     {df_reales['suma_ansiedad'].mean():.2f} ± {df_reales['suma_ansiedad'].std():.2f}
- Diferencia:       {diferencia_media:.2f} puntos

DISTRIBUCIÓN DE NIVELES:
                Sintéticos    Reales      Diferencia
Bajo:           {pct_sint[0]:5.1f}%      {pct_real[0]:5.1f}%     {diferencia_pct_bajo:5.1f}%
Medio:          {pct_sint[1]:5.1f}%      {pct_real[1]:5.1f}%     {diferencia_pct_medio:5.1f}%
Alto:           {pct_sint[2]:5.1f}%      {pct_real[2]:5.1f}%     {diferencia_pct_alto:5.1f}%

SIGNIFICANCIA ESTADÍSTICA:
- Distribuciones (KS):  {'DIFERENTES' if ks_pvalue < 0.05 else 'SIMILARES'} (p={ks_pvalue:.4f})
- Medias (t-test):      {'DIFERENTES' if t_pvalue < 0.05 else 'SIMILARES'} (p={t_pvalue:.4f})
- Proporciones (χ²):    {'DIFERENTES' if chi_pvalue < 0.05 else 'SIMILARES'} (p={chi_pvalue:.4f})

💡 INTERPRETACIÓN PARA TU TESIS:
""")

if ks_pvalue >= 0.05 and t_pvalue >= 0.05:
    print("✅ EXCELENTE: Los datos sintéticos replican bien las características de datos reales")
    print("   - Las distribuciones son estadísticamente similares")
    print("   - Puedes argumentar que tu generación sintética es válida")
elif diferencia_media < 2.0:
    print("✅ BUENO: Diferencias menores esperadas entre contextos (Bangladesh vs México)")
    print("   - La diferencia de medias es pequeña (<2 puntos)")
    print("   - Puedes usar esto para validar tu modelo")
else:
    print("⚠️  DIFERENCIAS NOTABLES: Considera el enfoque híbrido")
    print("   - Usa respuestas reales DASS-21 + contexto demográfico del ITO")

print(f"""
📁 ARCHIVOS GENERADOS EN '{output_dir}/':
- comparacion_completa.png (visualización principal)
- sinteticos_procesados.csv
- reales_procesados.csv
- tabla_comparativa.csv
- pruebas_estadisticas.txt

🎯 SIGUIENTE PASO:
Decide tu estrategia para la tesis:
1. Mantener datos sintéticos (si p > 0.05)
2. Usar enfoque híbrido (respuestas reales + demográficos ITO)
3. Entrenar modelo con ambos datasets para comparación
""")

print("="*80)
print("✅ COMPARACIÓN COMPLETADA")
print("="*80)