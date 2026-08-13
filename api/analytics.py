import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import mysql.connector
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

if __package__:
    from .config import DB_CONFIG, MODELO_PATH, format_db_config_public
else:
    from config import DB_CONFIG, MODELO_PATH, format_db_config_public


class AnxiTechAnalytics:

    def __init__(self):
        self.modelo = None
        self.feature_names = [
            'promedio_anterior',
            'semestre',
            'materias',
            'edad',
            'transporte',
            'familiares',
            'trabajo',
            'beca',
            'sexo',
            'estado_civil',
            'carrera',
            'maestros_estrictos',
            'tiene_hijos',
            'ingreso_mensual',
            'horas_sueno',
        ]
        self.cargar_modelo()

    def cargar_modelo(self):
        try:
            modelo_path = Path(MODELO_PATH)
            if not modelo_path.exists():
                print(f"ERROR: Archivo del modelo no encontrado en: {modelo_path}")
                self.modelo = None
                return
            self.modelo = joblib.load(str(modelo_path))
            print(f"Modelo cargado correctamente desde: {modelo_path}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.modelo = None

    def get_db_connection(self):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            print(f"Error de conexion a BD ({format_db_config_public()}): {type(e).__name__}: {e}")
            raise RuntimeError(
                "No se pudo conectar a la base de datos. "
                "Revisa las variables DB_* / MYSQL* / DATABASE_URL y la configuracion SSL."
            ) from e

    def get_estadisticas_generales(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = """
                SELECT
                    COUNT(*) as total,
                    CASE
                        WHEN SUM(valor) <= 4 THEN 'Bajo'
                        WHEN SUM(valor) <= 7 THEN 'Medio'
                        ELSE 'Alto'
                    END as nivel
                FROM alumno_pregunta
                WHERE id_pregunta IN (
                    SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                )
                GROUP BY id_alumno
            """
            cursor.execute(query)
            resultados = cursor.fetchall()

            distribucion = {'Bajo': 0, 'Medio': 0, 'Alto': 0}
            total_alumnos = 0

            for row in resultados:
                nivel = row['nivel']
                distribucion[nivel] += 1
                total_alumnos += 1

            porcentajes = {}
            if total_alumnos > 0:
                for nivel, cantidad in distribucion.items():
                    porcentajes[nivel] = round((cantidad / total_alumnos) * 100, 1)

            cursor.execute("""
                SELECT AVG(valor) as promedio
                FROM alumno_pregunta
                WHERE id_pregunta IN (
                    SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                )
            """)
            promedio_result = cursor.fetchone()
            ansiedad_promedio = round(promedio_result['promedio'] if promedio_result['promedio'] else 0, 2)

            return {
                'total_alumnos': total_alumnos,
                'distribucion': distribucion,
                'porcentajes': porcentajes,
                'ansiedad_promedio': ansiedad_promedio
            }
        finally:
            cursor.close()
            conn.close()

    def get_factores_riesgo(self) -> List[Dict]:
        if not self.modelo:
            return []

        importancias = self.modelo.feature_importances_

        descripciones = {
            'promedio_anterior': 'Rendimiento académico previo. Promedios bajos correlacionan con mayor ansiedad.',
            'edad': 'Edad del estudiante. Estudiantes más jóvenes tienden a mayor ansiedad.',
            'semestre': 'Semestre actual. Primeros semestres muestran mayor ansiedad por adaptación.',
            'materias': 'Carga académica. Mayor cantidad de materias aumenta el estrés.',
            'transporte': 'Medio de transporte. Transporte público largo aumenta estrés.',
            'familiares': 'Número de familiares. Mayor número puede aumentar presión económica.',
            'trabajo': 'Situación laboral. Trabajar mientras estudia aumenta presión.',
            'beca': 'Situación de beca. Presión por mantener promedio para conservar beca.',
            'sexo': 'Género del estudiante. Puede haber diferencias en manejo del estrés.',
            'estado_civil': 'Estado civil. Responsabilidades adicionales pueden aumentar estrés.',
            'carrera': 'Carrera específica. Algunas carreras tienen mayor carga de estrés.',
            'maestros_estrictos': 'Percepción de maestros estrictos. Genera presión académica adicional.',
            'tiene_hijos': 'Tener hijos. Responsabilidades familiares aumentan la carga de estrés.',
            'ingreso_mensual': 'Ingreso mensual. Bajos ingresos se asocian con mayor ansiedad económica.',
            'horas_sueno': 'Horas de sueño. Dormir menos de 6 horas aumenta significativamente la ansiedad.',
        }

        nombres_legibles = {
            'promedio_anterior': 'Promedio académico bajo',
            'edad': 'Edad joven (<21 años)',
            'semestre': 'Semestres iniciales (1-3)',
            'materias': 'Carga excesiva de materias (7+)',
            'transporte': 'Transporte público',
            'familiares': 'Número de familiares',
            'trabajo': 'Trabaja y estudia',
            'beca': 'Presión por mantener beca',
            'sexo': 'Género',
            'estado_civil': 'Estado civil',
            'carrera': 'Carrera específica',
            'maestros_estrictos': 'Maestros estrictos',
            'tiene_hijos': 'Tiene hijos',
            'ingreso_mensual': 'Ingreso mensual bajo',
            'horas_sueno': 'Pocas horas de sueño (<6h)',
        }

        factores = []
        for i, (nombre, importancia) in enumerate(zip(self.feature_names, importancias)):
            factores.append({
                'rank': i + 1,
                'nombre': nombres_legibles.get(nombre, nombre),
                'variable': nombre,
                'importancia': round(importancia, 3),
                'porcentaje_impacto': round(importancia * 100, 1),
                'descripcion': descripciones.get(nombre, '')
            })

        factores.sort(key=lambda x: x['importancia'], reverse=True)
        for i, factor in enumerate(factores):
            factor['rank'] = i + 1

        return factores[:5]

    def get_tabla_correlaciones(self) -> List[Dict]:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            condiciones_factores = [
                {'factor': 'Promedio académico bajo (<70)',  'condicion': 'promedio_anterior < 70'},
                {'factor': 'Carga excesiva de materias (7+)', 'condicion': 'materias >= 7'},
                {'factor': 'Trabaja y estudia',              'condicion': 'trabajo = 1'},
                {'factor': 'Transporte público', 'condicion': 'transporte = 0'},
                {'factor': 'Presión por mantener beca',      'condicion': 'beca = 1'},
                {'factor': 'Semestres iniciales (1-3)',       'condicion': 'semestre <= 3'},
                {'factor': 'Edad joven (<21 años)',           'condicion': 'edad < 21'},
                {'factor': 'Pocas horas de sueño (<6h)',      'condicion': 'horas_sueno < 6'},
{'factor': 'Ingreso mensual bajo (<3000)', 'condicion': 'ingreso_mensual <= 1'},
                {'factor': 'Maestros estrictos',              'condicion': 'maestros_estrictos = 1'},
                {'factor': 'Tiene hijos',                    'condicion': 'tiene_hijos = 1'},
            ]

            correlaciones = []
            for item in condiciones_factores:
                query = f"""
                    SELECT
                        nivel,
                        COUNT(*) as cantidad
                    FROM (
                        SELECT
                            c.id_alumno,
                            CASE
                                WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                                WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                                ELSE 'Alto'
                            END as nivel
                        FROM complemento c
                        JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                        WHERE {item['condicion']}
                        AND ap.id_pregunta IN (
                            SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                        )
                        GROUP BY c.id_alumno
                    ) AS niveles_por_alumno
                    GROUP BY nivel
                """
                cursor.execute(query)
                resultados = cursor.fetchall()

                distribucion = {'Bajo': 0, 'Medio': 0, 'Alto': 0}
                total = 0
                for row in resultados:
                    distribucion[row['nivel']] = row['cantidad']
                    total += row['cantidad']

                if total > 0:
                    bajo_pct  = round((distribucion['Bajo']  / total) * 100, 0)
                    medio_pct = round((distribucion['Medio'] / total) * 100, 0)
                    alto_pct  = round((distribucion['Alto']  / total) * 100, 0)
                    impacto   = round(((alto_pct * 3) + (medio_pct * 2) + (bajo_pct * 1)) / 3, 0)

                    correlaciones.append({
                        'factor': item['factor'],
                        'total_estudiantes': total,
                        'bajo': int(bajo_pct),
                        'medio': int(medio_pct),
                        'alto': int(alto_pct),
                        'impacto': int(impacto)
                    })

            correlaciones.sort(key=lambda x: x['impacto'], reverse=True)
            return correlaciones

        finally:
            cursor.close()
            conn.close()

    def get_correlacion_variable(self, variable: str, umbral: float) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = f"""
                SELECT
                    nivel,
                    COUNT(*) as cantidad
                FROM (
                    SELECT
                        c.id_alumno,
                        c.{variable},
                        CASE
                            WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                            WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                            ELSE 'Alto'
                        END as nivel
                    FROM complemento c
                    JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                    WHERE c.{variable} >= %s
                    AND ap.id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY c.id_alumno, c.{variable}
                ) AS niveles_por_alumno
                GROUP BY nivel
            """
            cursor.execute(query, (umbral,))
            resultados = cursor.fetchall()

            distribucion = {}
            total = 0
            for row in resultados:
                distribucion[row['nivel']] = row['cantidad']
                total += row['cantidad']

            distribucion_final = {}
            for nivel in ['Bajo', 'Medio', 'Alto']:
                cantidad = distribucion.get(nivel, 0)
                porcentaje = round((cantidad / total * 100), 1) if total > 0 else 0
                distribucion_final[nivel] = {'cantidad': cantidad, 'porcentaje': porcentaje}

            return {
                'variable': variable,
                'condicion': f'{variable} >= {umbral}',
                'total_alumnos_condicion': total,
                'distribucion_ansiedad': distribucion_final
            }
        finally:
            cursor.close()
            conn.close()

    def get_analisis_por_carrera(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = """
                SELECT
                    c.carrera,
                    c.id_alumno,
                    SUM(ap.valor) as suma_ansiedad,
                    CASE
                        WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                        WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                        ELSE 'Alto'
                    END as nivel
                FROM complemento c
                JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                WHERE ap.id_pregunta IN (
                    SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                )
                GROUP BY c.carrera, c.id_alumno
            """
            cursor.execute(query)
            resultados = cursor.fetchall()

            carreras_data = {}
            for row in resultados:
                carrera = row['carrera']
                if carrera not in carreras_data:
                    carreras_data[carrera] = {'total': 0, 'suma_total': 0, 'distribucion': {'Bajo': 0, 'Medio': 0, 'Alto': 0}}
                carreras_data[carrera]['total'] += 1
                carreras_data[carrera]['suma_total'] += row['suma_ansiedad']
                carreras_data[carrera]['distribucion'][row['nivel']] += 1

            carreras_list = []
            for carrera, data in carreras_data.items():
                total = data['total']
                ansiedad_prom = round(data['suma_total'] / total, 2) if total > 0 else 0
                porcentajes = {n: round((c / total * 100), 1) if total > 0 else 0 for n, c in data['distribucion'].items()}
                porcentaje_alto = porcentajes.get('Alto', 0)
                nivel_riesgo = 'Alto' if porcentaje_alto >= 20 else ('Medio' if porcentaje_alto >= 10 else 'Bajo')
                carreras_list.append({
                    'nombre': carrera,
                    'total_alumnos': total,
                    'ansiedad_promedio': ansiedad_prom,
                    'distribucion': data['distribucion'],
                    'porcentajes': porcentajes,
                    'nivel_riesgo': nivel_riesgo
                })

            carreras_list.sort(key=lambda x: x['ansiedad_promedio'], reverse=True)
            carrera_mayor_riesgo = carreras_list[0]['nombre'] if carreras_list else 'N/A'
            return {'carreras': carreras_list, 'carrera_mayor_riesgo': carrera_mayor_riesgo}
        finally:
            cursor.close()
            conn.close()

    def get_alertas(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM (
                    SELECT id_alumno
                    FROM alumno_pregunta
                    WHERE id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY id_alumno
                    HAVING SUM(valor) > 7
                ) AS alumnos_alto_riesgo
            """)
            resultado = cursor.fetchone()
            alumnos_riesgo_alto = resultado['total'] if resultado else 0
            cursor.fetchall()

            cursor.execute("SELECT COUNT(DISTINCT id_alumno) as total FROM alumno_pregunta")
            total_alumnos = cursor.fetchone()['total']
            porcentaje_riesgo = round((alumnos_riesgo_alto / total_alumnos * 100), 1) if total_alumnos > 0 else 0
            cursor.fetchall()

            cursor.execute("""
                SELECT c.carrera, COUNT(*) as cantidad_alto
                FROM complemento c
                JOIN (
                    SELECT id_alumno
                    FROM alumno_pregunta
                    WHERE id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY id_alumno
                    HAVING SUM(valor) > 7
                ) AS alto_riesgo ON c.id_alumno = alto_riesgo.id_alumno
                GROUP BY c.carrera
                ORDER BY cantidad_alto DESC
                LIMIT 1
            """)
            carrera_result = cursor.fetchone()

            if carrera_result:
                cursor.fetchall()
                cursor.execute("SELECT COUNT(*) as total_carrera FROM complemento WHERE carrera = %s", (carrera_result['carrera'],))
                total_carrera = cursor.fetchone()['total_carrera']
                porcentaje_carrera = round((carrera_result['cantidad_alto'] / total_carrera * 100), 1) if total_carrera > 0 else 0
                carrera_mayor_riesgo = {'nombre': carrera_result['carrera'], 'porcentaje': porcentaje_carrera}
            else:
                carrera_mayor_riesgo = {'nombre': 'N/A', 'porcentaje': 0}

            factores = self.get_factores_riesgo()
            factor_principal = factores[0]['nombre'] if factores else 'N/A'

            return {
                'alumnos_riesgo_alto': alumnos_riesgo_alto,
                'porcentaje_riesgo_alto': porcentaje_riesgo,
                'carrera_mayor_riesgo': carrera_mayor_riesgo,
                'semestre_critico': None,
                'factor_principal': factor_principal
            }
        finally:
            cursor.close()
            conn.close()

    def get_explicacion_individual(self, id_alumno: int) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("""
                SELECT
                    c.*,
                    SUM(ap.valor) as puntaje_ansiedad,
                    CASE
                        WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                        WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                        ELSE 'Alto'
                    END as nivel
                FROM complemento c
                JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                WHERE c.id_alumno = %s
                GROUP BY c.id_alumno
            """, (id_alumno,))

            alumno = cursor.fetchone()
            if not alumno:
                return {"error": "Alumno no encontrado"}

            factores_contribucion = []

            if self.modelo:
                importancias = self.modelo.feature_importances_

                for nombre, importancia in zip(self.feature_names, importancias):
                    valor_alumno = alumno.get(nombre)
                    if valor_alumno is None:
                        continue

                    cursor.fetchall()
                    cursor.execute(f"SELECT AVG({nombre}) as promedio FROM complemento")
                    row = cursor.fetchone()
                    promedio = row['promedio'] if row and row['promedio'] is not None else 0

                    es_riesgo = False
                    descripcion = ""

                    if nombre == 'promedio_anterior':
                        es_riesgo = valor_alumno < 80
                        descripcion = f"Promedio de {valor_alumno} (media: {promedio:.1f})"
                    elif nombre == 'materias':
                        es_riesgo = valor_alumno >= 7
                        descripcion = f"Cursando {valor_alumno} materias (media: {promedio:.1f})"
                    elif nombre == 'edad':
                        es_riesgo = valor_alumno < 21
                        descripcion = f"{valor_alumno} años (media: {promedio:.1f})"
                    elif nombre == 'horas_sueno':
                        es_riesgo = valor_alumno < 6
                        descripcion = f"{valor_alumno} horas de sueño (media: {promedio:.1f}h)"
                    elif nombre == 'ingreso_mensual':
                        es_riesgo = valor_alumno < 3000
                        descripcion = f"Ingreso ${valor_alumno} (media: ${promedio:.0f})"
                    elif nombre == 'maestros_estrictos':
                        es_riesgo = int(valor_alumno) == 1
                        descripcion = f"{'Tiene' if es_riesgo else 'No tiene'} maestros estrictos"
                    elif nombre == 'tiene_hijos':
                        es_riesgo = int(valor_alumno) == 1
                        descripcion = f"{'Tiene' if es_riesgo else 'No tiene'} hijos"

                    if es_riesgo:
                        factores_contribucion.append({
                            'factor': nombre,
                            'valor': valor_alumno,
                            'importancia': round(importancia * 100, 1),
                            'descripcion': descripcion,
                            'nivel_riesgo': 'Alto' if importancia > 0.15 else 'Medio'
                        })

                factores_contribucion.sort(key=lambda x: x['importancia'], reverse=True)

            cursor.fetchall()
            cursor.execute("""
                SELECT
                    p.pregunta,
                    ap.valor,
                    CASE ap.valor
                        WHEN 0 THEN 'No me aplicó'
                        WHEN 1 THEN 'Me aplicó un poco'
                        WHEN 2 THEN 'Me aplicó bastante'
                        WHEN 3 THEN 'Me aplicó mucho'
                    END as respuesta_texto
                FROM alumno_pregunta ap
                JOIN pregunta p ON ap.id_pregunta = p.id
                WHERE ap.id_alumno = %s
                AND p.categoria = 'ansiedad'
                ORDER BY ap.valor DESC
            """, (id_alumno,))
            respuestas_dass21 = cursor.fetchall()

            return {
                'alumno_id': id_alumno,
                'nivel_ansiedad': alumno['nivel'],
                'puntaje': round(alumno['puntaje_ansiedad'], 1),
                'factores_contribuyentes': factores_contribucion[:5],
                'respuestas_dass21': respuestas_dass21[:5],
                'perfil': {
                    'promedio': alumno['promedio_anterior'],
                    'materias': alumno['materias'],
                    'semestre': alumno['semestre'],
                    'carrera': alumno['carrera'],
                    'horas_sueno': alumno.get('horas_sueno'),
                    'ingreso_mensual': alumno.get('ingreso_mensual'),
                }
            }
        finally:
            cursor.close()
            conn.close()

    def get_tendencia_por_semestre(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = """
                SELECT
                    semestre,
                    nivel,
                    COUNT(*) as cantidad
                FROM (
                    SELECT
                        c.id_alumno,
                        c.semestre,
                        CASE
                            WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                            WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                            ELSE 'Alto'
                        END as nivel
                    FROM complemento c
                    JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                    WHERE ap.id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY c.id_alumno, c.semestre
                ) AS clasificacion_por_alumno
                GROUP BY semestre, nivel
                ORDER BY semestre, nivel
            """
            cursor.execute(query)
            resultados = cursor.fetchall()

            tendencias = {}
            for sem in range(1, 11):
                tendencias[sem] = {'Bajo': 0, 'Medio': 0, 'Alto': 0, 'total': 0}

            for row in resultados:
                sem = row['semestre']
                if sem in tendencias:
                    tendencias[sem][row['nivel']] += row['cantidad']
                    tendencias[sem]['total'] += row['cantidad']

            resultado_final = {}
            for sem, datos in tendencias.items():
                total = datos['total']
                if total > 0:
                    resultado_final[sem] = {
                        'total_estudiantes': total,
                        'porcentajes': {
                            'Bajo':  round((datos['Bajo']  / total) * 100, 1),
                            'Medio': round((datos['Medio'] / total) * 100, 1),
                            'Alto':  round((datos['Alto']  / total) * 100, 1)
                        },
                        'cantidades': {
                            'Bajo': datos['Bajo'], 'Medio': datos['Medio'], 'Alto': datos['Alto']
                        }
                    }

            if not resultado_final:
                return {'tendencias_por_semestre': {}, 'semestre_mas_critico': None, 'porcentaje_alto_critico': 0}

            semestre_critico = max(resultado_final.items(), key=lambda x: x[1]['porcentajes']['Alto'])

            return {
                'tendencias_por_semestre': resultado_final,
                'semestre_mas_critico': semestre_critico[0],
                'porcentaje_alto_critico': semestre_critico[1]['porcentajes']['Alto']
            }
        finally:
            cursor.close()
            conn.close()

    def get_evolucion_factor_por_semestre(self, factor: str) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            condiciones = {
                'promedio_anterior':  'c.promedio_anterior < 70',
                'materias':           'c.materias >= 7',
                'trabajo':            'c.trabajo = 1',
                'transporte':         "c.transporte = 'Transporte publico'",
                'beca':               'c.beca = 1',
                'semestre':           'c.semestre <= 3',
                'edad':               'c.edad < 21',
                'familiares':         'c.familiares >= 5',
                'sexo':               "c.sexo = 'F'",
                'estado_civil':       "c.estado_civil != 'Soltero'",
                'carrera':            "c.carrera = 'ISC'",
                'maestros_estrictos': 'c.maestros_estrictos = 1',
                'tiene_hijos':        'c.tiene_hijos = 1',
                'ingreso_mensual':    'c.ingreso_mensual < 3000',
                'horas_sueno':        'c.horas_sueno < 6',
            }

            condicion = condiciones.get(factor, 'c.promedio_anterior < 70')

            query = f"""
                SELECT
                    semestre,
                    nivel,
                    COUNT(*) as cantidad
                FROM (
                    SELECT
                        c.id_alumno,
                        c.semestre,
                        CASE
                            WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                            WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                            ELSE 'Alto'
                        END as nivel
                    FROM complemento c
                    JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                    WHERE {condicion}
                    AND ap.id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY c.id_alumno, c.semestre
                ) AS clasificacion
                GROUP BY semestre, nivel
                ORDER BY semestre, nivel
            """
            cursor.execute(query)
            resultados = cursor.fetchall()

            evolucion = {}
            for sem in range(1, 11):
                evolucion[sem] = {'Bajo': 0, 'Medio': 0, 'Alto': 0, 'total': 0}

            for row in resultados:
                sem = row['semestre']
                if sem in evolucion:
                    evolucion[sem][row['nivel']] += row['cantidad']
                    evolucion[sem]['total'] += row['cantidad']

            resultado = []
            for sem, datos in evolucion.items():
                total = datos['total']
                if total > 0:
                    resultado.append({
                        'semestre': sem,
                        'total': total,
                        'cantidades': {'Bajo': datos['Bajo'], 'Medio': datos['Medio'], 'Alto': datos['Alto']},
                        'porcentajes': {
                            'Bajo':  round((datos['Bajo']  / total) * 100, 1),
                            'Medio': round((datos['Medio'] / total) * 100, 1),
                            'Alto':  round((datos['Alto']  / total) * 100, 1)
                        }
                    })

            return {'factor': factor, 'evolucion': resultado}
        finally:
            cursor.close()
            conn.close()
    
    def get_metricas_modelo(self) -> Dict:
        """Entrena y evalúa un modelo nuevo con los datos reales de la BD."""
        conn = self.get_db_connection()

        try:
            query = """
            SELECT
                c.promedio_anterior, c.semestre, c.materias, c.edad,
                c.transporte, c.familiares, c.trabajo, c.beca,
                c.sexo, c.estado_civil, c.carrera,
                c.maestros_estrictos, c.tiene_hijos,
                c.ingreso_mensual, c.horas_sueno,
                SUM(ap.valor) as suma_ansiedad
            FROM complemento c
            INNER JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
            INNER JOIN pregunta p ON ap.id_pregunta = p.id
            WHERE p.categoria = 'ansiedad' AND p.status = 1
            GROUP BY c.id_alumno, c.promedio_anterior, c.semestre,
                     c.materias, c.edad, c.transporte, c.familiares,
                     c.trabajo, c.beca, c.sexo, c.estado_civil,
                     c.carrera, c.maestros_estrictos, c.tiene_hijos,
                     c.ingreso_mensual, c.horas_sueno
            HAVING COUNT(ap.id) = 7
            """

            df = pd.read_sql(query, conn)
            if len(df) == 0:
                return {"error": "No hay datos"}

            # ── 1. Limpiar tipos ──
            numericas = ['promedio_anterior', 'semestre', 'materias', 'edad',
                         'transporte', 'familiares', 'trabajo', 'beca',
                         'maestros_estrictos', 'tiene_hijos', 'ingreso_mensual',
                         'horas_sueno', 'suma_ansiedad']
            for col in numericas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

            df = df.dropna(subset=['suma_ansiedad'])

            # ── 2. Clasificar niveles ──
            niveles = []
            for s in df['suma_ansiedad'].values:
                if float(s) <= 4:
                    niveles.append('Bajo')
                elif float(s) <= 7:
                    niveles.append('Medio')
                else:
                    niveles.append('Alto')

            # ── 3. Preparar features ──
            feature_names = [
                 'promedio_anterior', 'semestre', 'materias', 'edad',
                'transporte', 'familiares', 'trabajo', 'beca',
                'sexo', 'estado_civil', 'carrera',
                'maestros_estrictos', 'tiene_hijos',
                'ingreso_mensual', 'horas_sueno',
            ]
            available = [f for f in feature_names if f in df.columns]

            X = df[available].copy()

            for col in ['sexo', 'estado_civil', 'carrera']:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            for col in X.columns:
                if col not in ['sexo', 'estado_civil', 'carrera']:
                    X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

            X = X.fillna(0)

            y = np.array(niveles)  # numpy array puro de strings

            # ── 4. División ──
            from collections import Counter
            counts = Counter(y)
            min_class = min(counts.values())
            use_stratify = min_class >= 3

            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y, test_size=0.2, random_state=42,
                stratify=y if use_stratify else None
            )

            # ── 5. Entrenar modelo NUEVO con datos reales ──
            from sklearn.ensemble import RandomForestClassifier as RFC
            modelo_nuevo = RFC(
                n_estimators=200, max_depth=4,
                min_samples_split=5, min_samples_leaf=3,
                max_features='sqrt', random_state=42,
                class_weight='balanced', n_jobs=-1
            )
            modelo_nuevo.fit(X_train, y_train)

            # ── 6. Predicciones ──
            y_pred_train = modelo_nuevo.predict(X_train)
            y_pred_test = modelo_nuevo.predict(X_test)

            train_acc = float(accuracy_score(y_train, y_pred_train))
            test_acc = float(accuracy_score(y_test, y_pred_test))
            prec_w = float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0))
            rec_w = float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0))
            f1_w = float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))
            f1_m = float(f1_score(y_test, y_pred_test, average='macro', zero_division=0))

            # ── 7. Cross-validation ──
            n_folds = min(5, min_class) if min_class >= 2 else 2
            try:
                cv = cross_val_score(modelo_nuevo, X.values, y, cv=n_folds, scoring='accuracy')
                cv_mean, cv_std = float(cv.mean()), float(cv.std())
                cv_folds = [float(s) for s in cv]
            except Exception:
                cv_mean, cv_std, cv_folds = test_acc, 0.0, []

            # ── 8. Report por clase ──
            report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
            metricas_clase = {}
            for nivel in ['Bajo', 'Medio', 'Alto']:
                if nivel in report:
                    metricas_clase[nivel] = {
                        'precision': round(report[nivel]['precision'], 4),
                        'recall': round(report[nivel]['recall'], 4),
                        'f1_score': round(report[nivel]['f1-score'], 4),
                        'support': int(report[nivel]['support'])
                    }

            # ── 9. Feature importance ──
            importancias = modelo_nuevo.feature_importances_
            dim_map = {
                'promedio_anterior': 'Académica', 'semestre': 'Académica',
    'materias': 'Académica', 'carrera': 'Académica',
    'beca': 'Académica', 'maestros_estrictos': 'Académica',
    'trabajo': 'Socioeconómica', 'transporte': 'Socioeconómica',
    'familiares': 'Socioeconómica', 'ingreso_mensual': 'Socioeconómica',
    'edad': 'Demográfica', 'sexo': 'Demográfica',
    'estado_civil': 'Demográfica', 'tiene_hijos': 'Demográfica',
    'horas_sueno': 'Demográfica',
            }
            fi = []
            for nombre, imp in zip(available, importancias):
                fi.append({
                    'variable': nombre,
                    'importancia': round(float(imp), 4),
                    'porcentaje': round(float(imp) * 100, 1),
                    'dimension': dim_map.get(nombre, 'Otra')
                })
            fi.sort(key=lambda x: x['importancia'], reverse=True)
            for i, f in enumerate(fi):
                f['rank'] = i + 1

            dim_acum = {}
            for f in fi:
                dim_acum[f['dimension']] = dim_acum.get(f['dimension'], 0) + f['importancia']
            dim_acum = {k: round(v * 100, 1) for k, v in
                        sorted(dim_acum.items(), key=lambda x: x[1], reverse=True)}

            # ── 10. Distribución ──
            dist = {}
            for nivel in ['Bajo', 'Medio', 'Alto']:
                c = int(np.sum(y == nivel))
                dist[nivel] = {'cantidad': c, 'porcentaje': round(c / len(y) * 100, 1)}

            # ── 11. Perfil descriptivo ──
            perfil = {
                'n_total': int(len(df)),
                'suma_media': round(float(df['suma_ansiedad'].mean()), 2),
                'suma_std': round(float(df['suma_ansiedad'].std()), 2),
                'suma_min': int(df['suma_ansiedad'].min()),
                'suma_max': int(df['suma_ansiedad'].max()),
            }
            if 'edad' in df.columns:
                perfil['edad_media'] = round(float(df['edad'].mean()), 1)
                perfil['edad_rango'] = f"{int(df['edad'].min())}-{int(df['edad'].max())}"
            if 'sexo' in df.columns:
                perfil['sexo'] = {str(k): int(v) for k, v in df['sexo'].value_counts().items()}
            if 'carrera' in df.columns:
                perfil['carreras'] = {str(k): int(v) for k, v in df['carrera'].value_counts().items()}
            if 'promedio_anterior' in df.columns:
                perfil['promedio_medio'] = round(float(df['promedio_anterior'].mean()), 1)
            if 'trabajo' in df.columns:
                perfil['pct_trabaja'] = round(float(pd.to_numeric(df['trabajo'], errors='coerce').mean()) * 100, 1)
            if 'beca' in df.columns:
                perfil['pct_beca'] = round(float(pd.to_numeric(df['beca'], errors='coerce').mean()) * 100, 1)

            return {
                'n_registros': int(len(df)),
                'n_features': len(available),
                'nota': 'Modelo entrenado y evaluado sobre datos reales (no usa modelo pre-cargado)',
                'distribucion': dist,
                'metricas': {
                    'accuracy_train': round(train_acc, 4),
                    'accuracy_test': round(test_acc, 4),
                    'precision_weighted': round(prec_w, 4),
                    'recall_weighted': round(rec_w, 4),
                    'f1_weighted': round(f1_w, 4),
                    'f1_macro': round(f1_m, 4),
                },
                'cross_validation': {
                    'n_folds': n_folds, 'media': round(cv_mean, 4),
                    'std': round(cv_std, 4), 'folds': cv_folds
                },
                'metricas_por_clase': metricas_clase,
                'feature_importance': fi,
                'importancia_por_dimension': dim_acum,
                'perfil_muestra': perfil,
                'config_modelo': {
                    'tipo': 'Random Forest (entrenado en datos reales)',
                    'n_estimators': 200,
                    'max_depth': 4,
                },
                'division': {
                    'train': int(len(X_train)),
                    'test': int(len(X_test)),
                },
                'timestamp': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            return {"error": str(e), "tipo": type(e).__name__, "trace": traceback.format_exc()}
        finally:
            conn.close()

    def get_sankey_data(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            factores_condiciones = [
                {'factor': 'Promedio bajo', 'condicion': 'c.promedio_anterior < 70'},
                {'factor': 'Materias 7+',   'condicion': 'c.materias >= 7'},
                {'factor': 'Trabaja',        'condicion': 'c.trabajo = 1'},
                {'factor': 'Transporte pub', 'condicion': 'c.transporte = 0'},
                {'factor': 'Beca',           'condicion': 'c.beca = 1'},
                {'factor': 'Sueño <6h',      'condicion': 'c.horas_sueno <= 1'},
                {'factor': 'Tiene hijos',    'condicion': 'c.tiene_hijos = 1'},
            ]

            nodes = []
            links = []

            factor_names = [f['factor'] for f in factores_condiciones]
            nivel_names = ['Bajo', 'Medio', 'Alto']

            for name in factor_names:
                nodes.append({'id': name, 'name': name})
            for name in nivel_names:
                nodes.append({'id': name, 'name': name})

            factor_idx = {f: i for i, f in enumerate(factor_names)}
            nivel_idx  = {n: len(factor_names) + i for i, n in enumerate(nivel_names)}

            for item in factores_condiciones:
                query = f"""
                    SELECT nivel, COUNT(*) as cantidad
                    FROM (
                        SELECT
                            c.id_alumno,
                            CASE
                                WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                                WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                                ELSE 'Alto'
                            END as nivel
                        FROM complemento c
                        JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                        WHERE {item['condicion']}
                        AND ap.id_pregunta IN (
                            SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                        )
                        GROUP BY c.id_alumno
                    ) AS subquery
                    GROUP BY nivel
                """
                cursor.execute(query)
                resultados = cursor.fetchall()

                for row in resultados:
                    if row['cantidad'] > 0:
                        links.append({
                            'source': factor_idx[item['factor']],
                            'target': nivel_idx[row['nivel']],
                            'value':  int(row['cantidad'])
                        })

            return {'nodes': nodes, 'links': links}

        finally:
            cursor.close()
            conn.close()


    def get_gauge_data(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("""
                SELECT nivel, COUNT(*) as cantidad
                FROM (
                    SELECT
                        id_alumno,
                        CASE
                            WHEN SUM(valor) <= 4 THEN 'Bajo'
                            WHEN SUM(valor) <= 7 THEN 'Medio'
                            ELSE 'Alto'
                        END as nivel
                    FROM alumno_pregunta
                    WHERE id_pregunta IN (
                        SELECT id FROM pregunta WHERE categoria = 'ansiedad' AND status = 1
                    )
                    GROUP BY id_alumno
                ) AS subquery
                GROUP BY nivel
            """)
            resultados = cursor.fetchall()

            distribucion = {'Bajo': 0, 'Medio': 0, 'Alto': 0}
            total = 0
            for row in resultados:
                distribucion[row['nivel']] = row['cantidad']
                total += row['cantidad']

            porcentaje_alto  = round((distribucion['Alto']  / total * 100), 1) if total > 0 else 0
            porcentaje_medio = round((distribucion['Medio'] / total * 100), 1) if total > 0 else 0

            if porcentaje_alto >= 30:
                nivel_riesgo = 'Alto'
                color = '#F44336'
            elif porcentaje_alto >= 15:
                nivel_riesgo = 'Moderado'
                color = '#FF9800'
            else:
                nivel_riesgo = 'Bajo'
                color = '#4CAF50'

            return {
                'porcentaje_alto':  porcentaje_alto,
                'porcentaje_medio': porcentaje_medio,
                'porcentaje_bajo':  round(100 - porcentaje_alto - porcentaje_medio, 1),
                'total_alumnos':    total,
                'nivel_riesgo':     nivel_riesgo,
                'color':            color,
                'distribucion':     distribucion
            }

        finally:
            cursor.close()
            conn.close()
    
    def get_por_institucion(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT 
                    institucion,
                    nivel_ansiedad,
                    COUNT(*) as total,
                    ROUND(AVG(promedio_avg), 1) as promedio_avg,
                    ROUND(AVG(materias_avg), 1) as materias_avg,
                    ROUND(AVG(sueno_avg), 1) as sueno_avg
                FROM (
                    SELECT 
                        c.institucion,
                        c.promedio_anterior as promedio_avg,
                        c.materias as materias_avg,
                        c.horas_sueno as sueno_avg,
                        CASE
                            WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                            WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                            ELSE 'Alto'
                        END as nivel_ansiedad
                    FROM complemento c
                    JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno
                    GROUP BY c.id_alumno, c.institucion, c.promedio_anterior,
                            c.materias, c.horas_sueno
                ) AS subquery
                GROUP BY institucion, nivel_ansiedad
                ORDER BY institucion, nivel_ansiedad
            """)
            resultados = cursor.fetchall()

            # Organizar por institución
            instituciones = {}
            for row in resultados:
                inst = row['institucion']
                if inst not in instituciones:
                    instituciones[inst] = {
                        'total': 0,
                        'niveles': {'Bajo': 0, 'Medio': 0, 'Alto': 0},
                        'promedio_avg': 0,
                        'materias_avg': 0,
                        'sueno_avg': 0,
                    }
                instituciones[inst]['niveles'][row['nivel_ansiedad']] = row['total']
                instituciones[inst]['total'] += row['total']
                instituciones[inst]['promedio_avg'] = row['promedio_avg']
                instituciones[inst]['materias_avg'] = row['materias_avg']
                instituciones[inst]['sueno_avg'] = row['sueno_avg']

            return {'instituciones': instituciones}

        finally:
            cursor.close()
            conn.close()
    
    def get_por_alumno(self) -> Dict:
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT 
                    u.id,
                    u.nombre,
                    u.apellido,
                    c.institucion,
                    c.carrera,
                    c.semestre,
                    c.promedio_anterior,
                    c.materias,
                    c.trabajo,
                    c.beca,
                    c.transporte,
                    c.horas_sueno,
                    c.maestros_estrictos,
                    c.ingreso_mensual,
                    c.tiene_hijos,
                    c.familiares,
                    SUM(ap.valor) as puntaje_dass21,
                    CASE
                        WHEN SUM(ap.valor) <= 4 THEN 'Bajo'
                        WHEN SUM(ap.valor) <= 7 THEN 'Medio'
                        ELSE 'Alto'
                    END as nivel_ansiedad
                FROM usuario u
                JOIN complemento c ON u.id = c.id_alumno
                JOIN alumno_pregunta ap ON u.id = ap.id_alumno
                GROUP BY u.id, u.nombre, u.apellido, c.institucion,
                        c.carrera, c.semestre, c.promedio_anterior,
                        c.materias, c.trabajo, c.beca, c.transporte,
                        c.horas_sueno, c.maestros_estrictos,
                        c.ingreso_mensual, c.tiene_hijos, c.familiares
                ORDER BY nivel_ansiedad DESC, u.apellido
            """)
            return {'alumnos': cursor.fetchall(), 'total': cursor.rowcount}
        finally:
            cursor.close()
            conn.close()

analytics = AnxiTechAnalytics()