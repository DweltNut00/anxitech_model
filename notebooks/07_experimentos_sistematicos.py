"""
================================================================
07_analisis_correlaciones.py - AnxiTech  (v2 - CORREGIDO)
================================================================
Métodos estadísticos:
  - Spearman       -> variables numéricas
  - Point-Biserial -> variables binarias (trabaja, beca)
  - Kruskal-Wallis -> variables categóricas
  - Eta-squared    -> tamaño del efecto categóricas

Autor : AnxiTech - Edgar J. Hernández-Andrade
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr, kruskal
import mysql.connector

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# FUNCIONES AUXILIARES  (definidas PRIMERO)
# ════════════════════════════════════════════

def _interpretar_rho(r_abs):
    if r_abs >= 0.70: return "Muy alta"
    if r_abs >= 0.50: return "Alta"
    if r_abs >= 0.30: return "Moderada"
    if r_abs >= 0.10: return "Baja"
    return "Despreciable"

def _interpretar_eta2(e):
    if e >= 0.14: return "Grande"
    if e >= 0.06: return "Mediano"
    if e >= 0.01: return "Pequeno"
    return "Despreciable"

# ════════════════════════════════════════════
# CONFIGURACION
# ════════════════════════════════════════════

DB_CONFIG = {
    'host':     'localhost',
    'user':     'root',
    'password': '',
    'database': 'anxitech'
}

OUTPUT_DIR = Path('reportes_correlacion')
OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════
# 1. EXTRACCION DE DATOS
# ════════════════════════════════════════════
print("=" * 65)
print("  ANALISIS DE CORRELACIONES - AnxiTech")
print("=" * 65)
print("\n[1/5] Conectando a la base de datos...")

try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    print("  OK Conexion exitosa")
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)

query = """
SELECT
    c.id_alumno,
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
    ROUND(AVG(ap.valor) * 7, 2) AS dass21_score,
    CASE
        WHEN AVG(ap.valor) * 7 <= 7  THEN 'Bajo'
        WHEN AVG(ap.valor) * 7 <= 14 THEN 'Medio'
        ELSE 'Alto'
    END AS nivel_ansiedad
FROM complemento c
JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
WHERE ap.id_pregunta IN (
    SELECT id FROM pregunta
    WHERE categoria = 'ansiedad' AND status = 1
)
GROUP BY
    c.id_alumno, c.promedio_anterior, c.semestre,
    c.materias, c.edad, c.transporte, c.familiares,
    c.trabajo, c.beca, c.sexo, c.estado_civil, c.carrera
HAVING COUNT(ap.valor) = 7
"""

try:
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    print(f"  OK Registros cargados: {len(df)}")
except Exception as e:
    print(f"  ERROR en consulta: {e}")
    cursor.close(); conn.close(); exit(1)

cursor.close()
conn.close()

if len(df) < 30:
    print(f"  AVISO: Solo {len(df)} registros. Resultados pueden no ser representativos.")

print(f"\n  Distribucion niveles de ansiedad:")
for nivel, cnt in df['nivel_ansiedad'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"     {nivel:5s}: {cnt:4d}  ({pct:.1f}%)")

# ════════════════════════════════════════════
# 2. VARIABLES POR TIPO
# ════════════════════════════════════════════
print("\n[2/5] Clasificando variables...")

VARS_NUMERICAS   = ['promedio_anterior', 'semestre', 'materias', 'edad']
VARS_BINARIAS    = ['trabajo', 'beca']
VARS_CATEGORICAS = ['transporte', 'familiares', 'sexo', 'estado_civil', 'carrera']
TARGET           = 'dass21_score'

# ════════════════════════════════════════════
# 3. CALCULOS DE CORRELACION
# ════════════════════════════════════════════
print("\n[3/5] Calculando correlaciones...\n")

resultados = []

# -- Spearman (numericas) --
print("  Numericas (Spearman rho):")
print("  " + "-" * 60)
for var in VARS_NUMERICAS:
    col = pd.to_numeric(df[var], errors='coerce')
    mask = col.notna() & df[TARGET].notna()
    x, y = col[mask].values, df[TARGET][mask].values
    if len(x) < 10:
        print(f"  {var:25s} -> datos insuficientes"); continue
    rho, p = spearmanr(x, y)
    intens = _interpretar_rho(abs(rho))
    resultados.append({'Variable': var, 'Tipo': 'Numerica', 'Metodo': 'Spearman',
                       'Estadistico': round(rho, 4), 'p_valor': round(p, 4),
                       'Significativo': 'Si' if p < 0.05 else 'No', 'Intensidad': intens})
    sig = "**" if p < 0.05 else "  "
    print(f"  {sig} {var:25s}  rho={rho:+.3f}   p={p:.4f}  [{intens}]")

# -- Point-Biserial (binarias) --
print("\n  Binarias (Point-Biserial r):")
print("  " + "-" * 60)
for var in VARS_BINARIAS:
    col = pd.to_numeric(df[var], errors='coerce')
    mask = col.notna() & df[TARGET].notna()
    # Forzar float64 para evitar object dtype en scipy
    x = col[mask].values.astype(float)
    y = df[TARGET][mask].values.astype(float)
    if len(x) < 10:
        print(f"  {var:25s} -> datos insuficientes"); continue
    r, p = pointbiserialr(x, y)
    intens = _interpretar_rho(abs(r))
    resultados.append({'Variable': var, 'Tipo': 'Binaria', 'Metodo': 'Point-Biserial',
                       'Estadistico': round(r, 4), 'p_valor': round(p, 4),
                       'Significativo': 'Si' if p < 0.05 else 'No', 'Intensidad': intens})
    sig = "**" if p < 0.05 else "  "
    print(f"  {sig} {var:25s}  r={r:+.3f}    p={p:.4f}  [{intens}]")

# -- Kruskal-Wallis (categoricas) --
print("\n  Categoricas (Kruskal-Wallis H + eta2):")
print("  " + "-" * 60)
for var in VARS_CATEGORICAS:
    grupos = [df.loc[df[var] == cat, TARGET].dropna().values
              for cat in df[var].dropna().unique()
              if len(df.loc[df[var] == cat, TARGET].dropna()) >= 2]
    if len(grupos) < 2:
        print(f"  {var:25s} -> menos de 2 grupos validos"); continue
    try:
        H, p = kruskal(*grupos)
        k = len(grupos)
        n = sum(len(g) for g in grupos)
        eta2 = max(0, (H - k + 1) / (n - k)) if n > k else 0
        intens = _interpretar_eta2(eta2)
        resultados.append({'Variable': var, 'Tipo': 'Categorica', 'Metodo': 'Kruskal-Wallis',
                           'Estadistico': round(H, 4), 'p_valor': round(p, 4),
                           'Significativo': 'Si' if p < 0.05 else 'No',
                           'Intensidad': f"{intens} (eta2={eta2:.3f})"})
        sig = "**" if p < 0.05 else "  "
        print(f"  {sig} {var:25s}  H={H:.3f}  p={p:.4f}  eta2={eta2:.3f}  [{intens}]")
    except Exception as e:
        print(f"  {var:25s} -> error: {e}")

# ════════════════════════════════════════════
# 4. TABLA RESUMEN + CSV
# ════════════════════════════════════════════
print("\n[4/5] Generando reporte...")

df_res = pd.DataFrame(resultados).sort_values('p_valor')

print("\n  TABLA RESUMEN")
print("  " + "=" * 85)
print(f"  {'Variable':<22} {'Tipo':<12} {'Metodo':<17} {'Estadistico':>12} {'p-valor':>8} {'Sig':>4}  Intensidad")
print("  " + "-" * 85)
for _, row in df_res.iterrows():
    sig_sym = "SI" if row['Significativo'] == 'Si' else "--"
    print(f"  {row['Variable']:<22} {row['Tipo']:<12} {row['Metodo']:<17} "
          f"{row['Estadistico']:>12.4f} {row['p_valor']:>8.4f} {sig_sym:>4}  {row['Intensidad']}")
print("  " + "=" * 85)

csv_path = OUTPUT_DIR / 'correlaciones_resultado.csv'
df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n  Guardado: {csv_path}")

# ════════════════════════════════════════════
# 5. GRAFICAS
# ════════════════════════════════════════════
print("\n[5/5] Generando graficas...")

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})

# Grafica 1: Heatmap + Barplot fuerza
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

num_cols = VARS_NUMERICAS + VARS_BINARIAS + [TARGET]
df_num = df[num_cols].apply(pd.to_numeric, errors='coerce').dropna()
if len(df_num) >= 10:
    corr_matrix = df_num.corr(method='spearman')
    mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask_tri, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                ax=axes[0], linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[0].set_title('Correlaciones Spearman\n(numericas y binarias)',
                      fontsize=12, fontweight='bold', pad=12)
    axes[0].tick_params(axis='x', rotation=30)

df_plot = df_res.copy()
df_plot['Fuerza'] = df_plot['Estadistico'].abs()
for i, row in df_plot.iterrows():
    if 'eta2=' in str(row['Intensidad']):
        try:
            eta2_val = float(str(row['Intensidad']).split('eta2=')[1].rstrip(')'))
            df_plot.at[i, 'Fuerza'] = eta2_val
        except Exception:
            pass

df_plot = df_plot.sort_values('Fuerza', ascending=True)
colores = ['#2ecc71' if r == 'Si' else '#e74c3c' for r in df_plot['Significativo']]
axes[1].barh(df_plot['Variable'], df_plot['Fuerza'], color=colores,
             edgecolor='white', height=0.6)
axes[1].set_xlabel('Fuerza de asociacion (|rho| o eta2)')
axes[1].set_title('Fuerza de asociacion con DASS-21\nverde=p<0.05 | rojo=no significativo',
                  fontsize=12, fontweight='bold', pad=12)
axes[1].axvline(x=0.10, color='orange', linestyle='--', linewidth=1, label='Umbral 0.10')
axes[1].legend(fontsize=9)

plt.tight_layout(pad=2.0)
g1 = OUTPUT_DIR / 'correlaciones_fuerza.png'
plt.savefig(g1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Grafica 1: {g1}")

# Grafica 2: Boxplots categoricas
n_cats = len(VARS_CATEGORICAS)
fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 5))
if n_cats == 1: axes = [axes]

for ax, var in zip(axes, VARS_CATEGORICAS):
    datos = df[[var, TARGET]].dropna()
    orden = (datos.groupby(var)[TARGET].median()
             .sort_values(ascending=False).index.tolist())
    sns.boxplot(data=datos, x=var, y=TARGET, order=orden,
                palette='husl', ax=ax, width=0.55,
                flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('DASS-21 Score' if ax == axes[0] else '')
    ax.tick_params(axis='x', rotation=30)
    ax.axhline(y=14, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.suptitle('DASS-21 por variable categorica', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
g2 = OUTPUT_DIR / 'boxplots_categoricas.png'
plt.savefig(g2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Grafica 2: {g2}")

# Grafica 3: Scatter numericas
fig, axes = plt.subplots(1, len(VARS_NUMERICAS), figsize=(14, 4))
for ax, var in zip(axes, VARS_NUMERICAS):
    x = pd.to_numeric(df[var], errors='coerce')
    y_val = df[TARGET]
    mask = x.notna() & y_val.notna()
    ax.scatter(x[mask], y_val[mask], alpha=0.35, s=25,
               color='#3498db', edgecolors='none')
    try:
        z = np.polyfit(x[mask], y_val[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color='#e74c3c', linewidth=1.5)
    except Exception:
        pass
    rho, p_v = spearmanr(x[mask], y_val[mask])
    ax.set_title(f'{var}\nrho={rho:+.2f}, p={p_v:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('DASS-21 Score' if ax == axes[0] else '')
    ax.axhline(y=14, color='orange', linestyle='--', linewidth=1, alpha=0.7)

plt.suptitle('Scatter: variables numericas vs DASS-21', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
g3 = OUTPUT_DIR / 'scatter_numericas.png'
plt.savefig(g3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Grafica 3: {g3}")

# ════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════
sig_vars = df_res[df_res['Significativo'] == 'Si']['Variable'].tolist()
no_sig   = df_res[df_res['Significativo'] == 'No']['Variable'].tolist()

print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"\n  Total variables: {len(df_res)}")
print(f"  Significativas (p<0.05): {len(sig_vars)}")
for v in sig_vars:
    row = df_res[df_res['Variable'] == v].iloc[0]
    print(f"     SI  {v:25s}  p={row['p_valor']:.4f}  [{row['Intensidad']}]")
print(f"\n  No significativas: {len(no_sig)}")
for v in no_sig:
    row = df_res[df_res['Variable'] == v].iloc[0]
    print(f"     --  {v:25s}  p={row['p_valor']:.4f}")

print(f"""
  INTERPRETACION PARA TESIS:
  - Variables significativas: justifican inclusion en Random Forest.
  - Variables no significativas: argumenta cobertura multidimensional
    del schema como contribucion teorica del capitulo 3.

  Archivos en '{OUTPUT_DIR}/':
    correlaciones_resultado.csv
    correlaciones_fuerza.png
    boxplots_categoricas.png
    scatter_numericas.png
""")
print("=" * 65)
print("  COMPLETADO")
print("=" * 65)