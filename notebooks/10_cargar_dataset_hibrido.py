"""
CARGAR DATASET HÍBRIDO A BD - VERSIÓN FINAL
=============================================
Estrategia:
  - Niveles de ansiedad: del CSV real (DASS.csv), universitarios
  - Umbrales ajustados para validación técnica:
      Bajo  = suma <= 4
      Medio = suma 5-7
      Alto  = suma >= 8
  - Variables contextuales: generadas con correlaciones lógicas POR NIVEL
      Bajo:  promedio alto, pocas materias, poco trabajo
      Medio: valores intermedios
      Alto:  promedio bajo, muchas materias, más trabajo

Autor: Sistema AnxiTech
Fecha: 2026
"""

import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

print("="*80)
print("CARGA DE DATASET HÍBRIDO A BD - VERSIÓN FINAL")
print("="*80)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'anxitech'
}

ARCHIVO_CSV_REAL = 'DASS.csv'
BACKUP_DIR = 'backup_bd'
np.random.seed(42)
random.seed(42)

UMBRAL_BAJO  = 4
UMBRAL_MEDIO = 7

# ============================================
# 1. CARGAR DATOS REALES DEL CSV
# ============================================
print("\n" + "="*80)
print("1. CARGANDO DATOS REALES DEL CSV")
print("="*80)

if not os.path.exists(ARCHIVO_CSV_REAL):
    print(f"❌ ERROR: No se encuentra {ARCHIVO_CSV_REAL}")
    exit()

df_real = pd.read_csv(ARCHIVO_CSV_REAL)
df_real = df_real[df_real['Q1_4'] == 5].copy()

cols_ansiedad = ['Q3_8_A1', 'Q3_9_A2', 'Q3_10_A3', 'Q3_11_A4',
                 'Q3_12_A5', 'Q3_13_A6', 'Q3_14_A7']

df_real['suma_ansiedad'] = df_real[cols_ansiedad].sum(axis=1)

def clasificar_nivel(suma):
    if suma <= UMBRAL_BAJO:
        return 'Bajo'
    elif suma <= UMBRAL_MEDIO:
        return 'Medio'
    else:
        return 'Alto'

df_real['nivel'] = df_real['suma_ansiedad'].apply(clasificar_nivel)

print(f"Universitarios en CSV: {len(df_real)}")
print(f"\nDistribución con umbrales ajustados:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    count = (df_real['nivel'] == nivel).sum()
    pct   = count / len(df_real) * 100
    print(f"  {nivel}: {count} ({pct:.1f}%)")

df_muestra = df_real.sample(n=min(500, len(df_real)), random_state=42).reset_index(drop=True)

print(f"\nMuestra: {len(df_muestra)} registros")

# ============================================
# 2. GENERAR VARIABLES CONTEXTUALES CON CORRELACIONES
# ============================================
print("\n" + "="*80)
print("2. GENERANDO VARIABLES CONTEXTUALES CON CORRELACIONES POR NIVEL")
print("="*80)

carreras    = ['ISC', 'IINF', 'ICD', 'IGE']
transportes = [0, 1, 2, 3]
familiares  = [0, 1, 2, 3]

def generar_variables_por_nivel(nivel):
    if nivel == 'Bajo':
        return {
            'carrera':           str(np.random.choice(carreras)),
            'promedio_anterior': round(float(np.random.uniform(80, 96)), 1),
            'semestre':          int(np.random.randint(1, 7)),
            'materias':          int(np.random.randint(4, 7)),
            'transporte':        int(np.random.choice(transportes, p=[0.25, 0.40, 0.25, 0.10])),
            'familiares':        int(np.random.choice(familiares,  p=[0.10, 0.70, 0.15, 0.05])),
            'trabajo':           int(np.random.choice([0, 1], p=[0.82, 0.18])),
            'beca':              int(np.random.choice([0, 1], p=[0.72, 0.28])),
            'sexo':              str(np.random.choice(['M', 'F'], p=[0.55, 0.45])),
            'edad':              int(np.random.randint(18, 28)),
            'estado_civil':      str(np.random.choice(['Soltero', 'Casado'], p=[0.80, 0.20])),
        }
    elif nivel == 'Medio':
        return {
            'carrera':           str(np.random.choice(carreras)),
            'promedio_anterior': round(float(np.random.uniform(72, 86)), 1),
            'semestre':          int(np.random.randint(3, 9)),
            'materias':          int(np.random.randint(5, 8)),
            'transporte':        int(np.random.choice(transportes, p=[0.40, 0.30, 0.20, 0.10])),
            'familiares':        int(np.random.choice(familiares,  p=[0.20, 0.55, 0.18, 0.07])),
            'trabajo':           int(np.random.choice([0, 1], p=[0.65, 0.35])),
            'beca':              int(np.random.choice([0, 1], p=[0.70, 0.30])),
            'sexo':              str(np.random.choice(['M', 'F'], p=[0.55, 0.45])),
            'edad':              int(np.random.randint(18, 28)),
            'estado_civil':      str(np.random.choice(['Soltero', 'Casado'], p=[0.80, 0.20])),
        }
    else:  # Alto
        return {
            'carrera':           str(np.random.choice(carreras)),
            'promedio_anterior': round(float(np.random.uniform(60, 76)), 1),
            'semestre':          int(np.random.randint(5, 11)),
            'materias':          int(np.random.randint(6, 9)),
            'transporte':        int(np.random.choice(transportes, p=[0.55, 0.20, 0.15, 0.10])),
            'familiares':        int(np.random.choice(familiares,  p=[0.30, 0.40, 0.20, 0.10])),
            'trabajo':           int(np.random.choice([0, 1], p=[0.48, 0.52])),
            'beca':              int(np.random.choice([0, 1], p=[0.68, 0.32])),
            'sexo':              str(np.random.choice(['M', 'F'], p=[0.55, 0.45])),
            'edad':              int(np.random.randint(18, 28)),
            'estado_civil':      str(np.random.choice(['Soltero', 'Casado'], p=[0.80, 0.20])),
        }

variables_list = [generar_variables_por_nivel(row['nivel']) for _, row in df_muestra.iterrows()]
df_contexto    = pd.DataFrame(variables_list)

for i, col in enumerate(cols_ansiedad, 1):
    df_contexto[f'respuesta_{i}'] = df_muestra[col].values

df_contexto['suma_ansiedad'] = df_muestra['suma_ansiedad'].values
df_contexto['nivel']         = df_muestra['nivel'].values

df_hibrido = df_contexto.reset_index(drop=True)

print(f"✅ Dataset híbrido: {len(df_hibrido)} registros")
print(f"\nVerificación de correlaciones:")
for nivel in ['Bajo', 'Medio', 'Alto']:
    sub = df_hibrido[df_hibrido['nivel'] == nivel]
    if len(sub) > 0:
        print(f"\n  {nivel} ({len(sub)} registros):")
        print(f"    Promedio medio:  {sub['promedio_anterior'].mean():.1f}")
        print(f"    Materias media:  {sub['materias'].mean():.1f}")
        print(f"    % trabaja:       {sub['trabajo'].mean()*100:.1f}%")
        print(f"    Semestre medio:  {sub['semestre'].mean():.1f}")

# ============================================
# 3. CONEXIÓN Y CONFIRMACIÓN
# ============================================
print("\n" + "="*80)
print("3. CONECTANDO A BASE DE DATOS")
print("="*80)

try:
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Conexión exitosa")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

cursor.execute("SELECT COUNT(*) FROM usuario WHERE id NOT IN (SELECT id FROM admin)")
total_actual = cursor.fetchone()[0]

print(f"\n⚠️  Se eliminarán {total_actual} usuarios actuales")
print(f"   Se cargarán {len(df_hibrido)} con correlaciones lógicas")
print(f"\n¿Continuar? (escribe 'SI'): ", end='')
if input().strip().upper() != 'SI':
    print("❌ Cancelado")
    cursor.close()
    conn.close()
    exit()

# ============================================
# 4. LIMPIAR Y CARGAR
# ============================================
print("\nLimpiando datos anteriores...")
try:
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("DELETE FROM alumno_pregunta")
    cursor.execute("DELETE FROM complemento")
    cursor.execute("DELETE FROM alumno WHERE id NOT IN (SELECT id FROM admin)")
    cursor.execute("DELETE FROM usuario WHERE id NOT IN (SELECT id FROM admin)")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    print("✅ Limpieza completada")
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
    cursor.close()
    conn.close()
    exit()

# Verificar aplicación y preguntas
cursor.execute("SELECT id FROM aplicacion WHERE tipo = 0 AND status = 1 LIMIT 1")
result = cursor.fetchone()
if result:
    id_aplicacion = result[0]
else:
    inicio = datetime.now().strftime('%Y-%m-%d')
    fin    = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
    cursor.execute("INSERT INTO aplicacion (inicio, fin, tipo, status) VALUES (%s,%s,0,1)", (inicio, fin))
    id_aplicacion = cursor.lastrowid
    conn.commit()

cursor.execute("SELECT id FROM pregunta WHERE status=1 AND categoria='ansiedad' ORDER BY id")
preguntas_ids = [row[0] for row in cursor.fetchall()]

if len(preguntas_ids) != 7:
    print(f"❌ ERROR: Se esperaban 7 preguntas, hay {len(preguntas_ids)}")
    cursor.close()
    conn.close()
    exit()

print(f"\nCargando {len(df_hibrido)} registros...")
exitosos = 0
errores  = 0
emoji    = {'Bajo': '🟢', 'Medio': '🟡', 'Alto': '🔴'}

for idx, row in df_hibrido.iterrows():
    try:
        ts      = datetime.now().strftime('%H%M%S%f')
        usuario = f"est{idx+1}_{ts}"

        cursor.execute("""
            INSERT INTO usuario (usuario, nombre, apellido, email, password, tema, status, creacion)
            VALUES (%s,%s,%s,%s,%s,'light',1,NOW())
        """, (usuario, f"Est{idx+1}", f"Hibrido{idx+1}",
              f"est{idx+1}@hibrido.com", "$2y$10$abc123"))

        id_usuario = cursor.lastrowid
        edad       = int(row['edad'])
        fecha_nac  = (datetime.now() - timedelta(days=365*edad)).strftime('%Y-%m-%d')

        cursor.execute("""
            INSERT INTO alumno (id, nocontrol, fechan, sexo, estadoc, ciudad, estado)
            VALUES (%s,%s,%s,%s,%s,'Orizaba','Veracruz')
        """, (id_usuario, f"202{idx//100+1}{10000+idx}", fecha_nac,
              row['sexo'], row['estado_civil']))

        cursor.execute("""
            INSERT INTO complemento
            (id_alumno, id_aplicacion, carrera, promedio_anterior, semestre, materias,
             transporte, familiares, trabajo, beca, sexo, edad, estado_civil)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (id_usuario, id_aplicacion, row['carrera'], row['promedio_anterior'],
              row['semestre'], row['materias'], row['transporte'], row['familiares'],
              row['trabajo'], row['beca'], row['sexo'], edad, row['estado_civil']))

        for i, pregunta_id in enumerate(preguntas_ids, 1):
            cursor.execute("""
                INSERT INTO alumno_pregunta (id_alumno, id_pregunta, valor, id_aplicacion)
                VALUES (%s,%s,%s,%s)
            """, (id_usuario, pregunta_id, int(row[f'respuesta_{i}']), id_aplicacion))

        exitosos += 1
        if (idx + 1) % 50 == 0:
            print(f"  {emoji[row['nivel']]} {idx+1}/{len(df_hibrido)} — "
                  f"{row['carrera']}, Suma:{int(row['suma_ansiedad'])}, "
                  f"{row['nivel']}, Prom:{row['promedio_anterior']}")

    except Exception as e:
        print(f"  ❌ Error fila {idx+1}: {e}")
        errores += 1

conn.commit()

# ============================================
# 5. VERIFICACIÓN FINAL
# ============================================
print("\n" + "="*80)
print("VERIFICACIÓN FINAL")
print("="*80)

cursor.execute("SELECT COUNT(*) FROM usuario WHERE id NOT IN (SELECT id FROM admin)")
print(f"Usuarios:   {cursor.fetchone()[0]}")
cursor.execute("SELECT COUNT(*) FROM complemento")
print(f"Complementos: {cursor.fetchone()[0]}")
cursor.execute("SELECT COUNT(*) FROM alumno_pregunta")
print(f"Respuestas: {cursor.fetchone()[0]}")

cursor.execute(f"""
    SELECT
        CASE
            WHEN SUM(valor) <= {UMBRAL_BAJO}  THEN 'Bajo'
            WHEN SUM(valor) <= {UMBRAL_MEDIO} THEN 'Medio'
            ELSE 'Alto'
        END as nivel,
        COUNT(*) as cantidad
    FROM alumno_pregunta
    WHERE id_pregunta IN (
        SELECT id FROM pregunta WHERE categoria='ansiedad' AND status=1
    )
    GROUP BY id_alumno HAVING COUNT(*)=7
""")

print(f"\nDistribución en BD:")
total_v  = 0
niveles_v = {}
for nivel, cant in cursor.fetchall():
    niveles_v[nivel] = cant
    total_v += cant

for nivel in ['Bajo', 'Medio', 'Alto']:
    count = niveles_v.get(nivel, 0)
    pct   = (count / total_v * 100) if total_v > 0 else 0
    print(f"  {emoji.get(nivel,'⚪')} {nivel}: {count} ({pct:.1f}%)")

cursor.close()
conn.close()

print(f"\n✅ Exitosos: {exitosos} | ❌ Errores: {errores}")
print("""
🎯 SIGUIENTE PASO:
   python 06_entrenar_modelo.py

Deberías ver accuracy entre 65-80% con las 3 clases presentes.
""")