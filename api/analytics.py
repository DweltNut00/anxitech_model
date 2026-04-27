from pathlib import Path
from typing import Dict, List

import joblib
import mysql.connector
import numpy as np
import pandas as pd

if __package__:
    from .config import DB_CONFIG, MODELO_PATH, format_db_config_public
else:
    from config import DB_CONFIG, MODELO_PATH, format_db_config_public

class AnxiTechAnalytics:
    
    def __init__(self):
        """Inicializa el módulo de análisis"""
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
            'carrera'
        ]
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Carga el modelo de ML entrenado"""
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
        """Crea conexión a la base de datos"""
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
        """Obtiene estadísticas generales del sistema"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Query CORREGIDA: usar SUM(valor) en lugar de AVG(valor)*7
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
            
            # Procesar resultados
            distribucion = {'Bajo': 0, 'Medio': 0, 'Alto': 0}
            total_alumnos = 0
            
            for row in resultados:
                nivel = row['nivel']
                distribucion[nivel] += 1
                total_alumnos += 1
            
            # Calcular porcentajes
            porcentajes = {}
            if total_alumnos > 0:
                for nivel, cantidad in distribucion.items():
                    porcentajes[nivel] = round((cantidad / total_alumnos) * 100, 1)
            
            # Promedio de ansiedad
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
        """Obtiene los factores de riesgo basados en el modelo"""
        if not self.modelo:
            return []
        
        importancias = self.modelo.feature_importances_
        
        descripciones = {
            'promedio_anterior': 'Rendimiento académico previo. Promedios bajos correlacionan con mayor ansiedad.',
            'edad': 'Edad del estudiante. Estudiantes más jóvenes tienden a mayor ansiedad.',
            'semestre': 'Semestre actual. Primeros semestres muestran mayor ansiedad por adaptación.',
            'materias': 'Carga académica. Mayor cantidad de materias aumenta el estrés.',
            'transporte': 'Medio de transporte. Transporte público largo aumenta estrés.',
            'familiares': 'Número de familiares. Mayor número puede aumentar presión económica.',  # ← AGREGAR
            'trabajo': 'Situación laboral. Trabajar mientras estudia aumenta presión.',
            'beca': 'Situación de beca. Presión por mantener promedio para conservar beca.',
            'sexo': 'Género del estudiante. Puede haber diferencias en manejo del estrés.',  # ← AGREGAR
            'estado_civil': 'Estado civil. Responsabilidades adicionales pueden aumentar estrés.',  # ← AGREGAR
            'carrera': 'Carrera específica. Algunas carreras tienen mayor carga de estrés.'  # ← AGREGAR
        }
        
        nombres_legibles = {
            'promedio_anterior': 'Promedio académico bajo',
            'edad': 'Edad joven (<21 años)',
            'semestre': 'Semestres iniciales (1-3)',
            'materias': 'Carga excesiva de materias (7+)',
            'transporte': 'Transporte público',
            'familiares': 'Número de familiares',  # ← AGREGAR
            'trabajo': 'Trabaja y estudia',
            'beca': 'Presión por mantener beca',
            'sexo': 'Género',  # ← AGREGAR
            'estado_civil': 'Estado civil',  # ← AGREGAR
            'carrera': 'Carrera específica'  # ← AGREGAR
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
        """
        Obtiene la tabla completa de correlaciones entre factores y niveles de ansiedad
        """
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            correlaciones = []
            
            # Definir condiciones para cada factor
            condiciones_factores = [
                {
                    'factor': 'Promedio académico bajo (<70)',
                    'condicion': 'promedio_anterior < 70'
                },
                {
                    'factor': 'Carga excesiva de materias (7+)',
                    'condicion': 'materias >= 7'
                },
                {
                    'factor': 'Trabaja y estudia',
                    'condicion': 'trabajo = 1'
                },
                {
                    'factor': 'Transporte público',
                    'condicion': "transporte = 'Transporte publico'"
                },
                {
                    'factor': 'Presión por mantener beca',
                    'condicion': 'beca = 1'
                },
                {
                    'factor': 'Semestres iniciales (1-3)',
                    'condicion': 'semestre <= 3'
                },
                {
                    'factor': 'Edad joven (<21 años)',
                    'condicion': 'edad < 21'
                }
            ]
            
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
                
                # Procesar distribución
                distribucion = {'Bajo': 0, 'Medio': 0, 'Alto': 0}
                total = 0
                
                for row in resultados:
                    nivel = row['nivel']
                    cantidad = row['cantidad']
                    distribucion[nivel] = cantidad
                    total += cantidad
                
                # Calcular porcentajes
                if total > 0:
                    bajo_pct = round((distribucion['Bajo'] / total) * 100, 0)
                    medio_pct = round((distribucion['Medio'] / total) * 100, 0)
                    alto_pct = round((distribucion['Alto'] / total) * 100, 0)
                    
                    # Impacto ponderado: Alto=3, Medio=2, Bajo=1
                    impacto = round(((alto_pct * 3) + (medio_pct * 2) + (bajo_pct * 1)) / 3, 0)
                    
                    correlaciones.append({
                        'factor': item['factor'],
                        'total_estudiantes': total,  # ← AGREGAR ESTA LÍNEA
                        'bajo': int(bajo_pct),
                        'medio': int(medio_pct),
                        'alto': int(alto_pct),
                        'impacto': int(impacto)
                    })
            
            # Ordenar por impacto
            correlaciones.sort(key=lambda x: x['impacto'], reverse=True)
            
            return correlaciones
            
        finally:
            cursor.close()
            conn.close()
    
    def get_correlacion_variable(self, variable: str, umbral: float) -> Dict:
        """Analiza correlación entre una variable específica y el nivel de ansiedad"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Query CORREGIDA: usar SUM(valor)
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
                nivel = row['nivel']
                cantidad = row['cantidad']
                distribucion[nivel] = cantidad
                total += cantidad
            
            distribucion_final = {}
            for nivel in ['Bajo', 'Medio', 'Alto']:
                cantidad = distribucion.get(nivel, 0)
                porcentaje = round((cantidad / total * 100), 1) if total > 0 else 0
                distribucion_final[nivel] = {
                    'cantidad': cantidad,
                    'porcentaje': porcentaje
                }
            
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
        """Analiza niveles de ansiedad por carrera"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Query CORREGIDA: usar SUM(valor)
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
                nivel = row['nivel']
                suma = row['suma_ansiedad']
                
                if carrera not in carreras_data:
                    carreras_data[carrera] = {
                        'total': 0,
                        'suma_total': 0,
                        'distribucion': {'Bajo': 0, 'Medio': 0, 'Alto': 0}
                    }
                
                carreras_data[carrera]['total'] += 1
                carreras_data[carrera]['suma_total'] += suma
                carreras_data[carrera]['distribucion'][nivel] += 1
            
            carreras_list = []
            for carrera, data in carreras_data.items():
                total = data['total']
                ansiedad_prom = round(data['suma_total'] / total, 2) if total > 0 else 0
                
                porcentajes = {}
                for nivel, cantidad in data['distribucion'].items():
                    porcentajes[nivel] = round((cantidad / total * 100), 1) if total > 0 else 0
                
                porcentaje_alto = porcentajes.get('Alto', 0)
                if porcentaje_alto >= 20:
                    nivel_riesgo = 'Alto'
                elif porcentaje_alto >= 10:
                    nivel_riesgo = 'Medio'
                else:
                    nivel_riesgo = 'Bajo'
                
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
            
            return {
                'carreras': carreras_list,
                'carrera_mayor_riesgo': carrera_mayor_riesgo
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def get_alertas(self) -> Dict:
        """Obtiene alertas tempranas del sistema"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
    
        try:
            # Query CORREGIDA: SUM(valor) > 7 para nivel Alto
            query_riesgo = """
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
            """
            cursor.execute(query_riesgo)
            resultado = cursor.fetchone()
            alumnos_riesgo_alto = resultado['total'] if resultado else 0
            
            cursor.fetchall()
            
            cursor.execute("""
                SELECT COUNT(DISTINCT id_alumno) as total
                FROM alumno_pregunta
            """)
            
            total_alumnos = cursor.fetchone()['total']
            porcentaje_riesgo = round((alumnos_riesgo_alto / total_alumnos * 100), 1) if total_alumnos > 0 else 0
            
            cursor.fetchall()
            
            # Query CORREGIDA para carrera con mayor riesgo
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
                cursor.execute("""
                    SELECT COUNT(*) as total_carrera
                    FROM complemento
                    WHERE carrera = %s
                """, (carrera_result['carrera'],))
                
                total_carrera = cursor.fetchone()['total_carrera']
                porcentaje_carrera = round((carrera_result['cantidad_alto'] / total_carrera * 100), 1) if total_carrera > 0 else 0
                
                carrera_mayor_riesgo = {
                    'nombre': carrera_result['carrera'],
                    'porcentaje': porcentaje_carrera
                }
            else:
                carrera_mayor_riesgo = {
                    'nombre': 'N/A',
                    'porcentaje': 0
                }
            
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
        """Explica por qué un alumno tiene determinado nivel de ansiedad"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Query CORREGIDA: usar SUM(valor)
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
                feature_names = self.feature_names
                
                for nombre, importancia in zip(feature_names, importancias):
                    valor_alumno = alumno[nombre]
                    
                    cursor.fetchall()
                    cursor.execute(f"""
                        SELECT AVG({nombre}) as promedio
                        FROM complemento
                    """)
                    promedio = cursor.fetchone()['promedio']
                    
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
                    'carrera': alumno['carrera']
                }
            }
            
        finally:
            cursor.close()
            conn.close()
    # AGREGAR A analytics.py

    def get_tendencia_por_semestre(self) -> Dict:
        """Analiza evolución de ansiedad por semestre"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Query CORREGIDA: cuenta estudiantes ÚNICOS por semestre
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
            
            # Inicializar estructura
            tendencias = {}
            for sem in range(1, 11):
                tendencias[sem] = {'Bajo': 0, 'Medio': 0, 'Alto': 0, 'total': 0}
            
            # Contar por semestre y nivel
            for row in resultados:
                sem = row['semestre']
                if sem in tendencias:
                    nivel = row['nivel']
                    cantidad = row['cantidad']
                    tendencias[sem][nivel] += cantidad
                    tendencias[sem]['total'] += cantidad
            
            # Convertir a porcentajes
            resultado_final = {}
            for sem, datos in tendencias.items():
                total = datos['total']
                if total > 0:
                    resultado_final[sem] = {
                        'total_estudiantes': total,
                        'porcentajes': {
                            'Bajo': round((datos['Bajo'] / total) * 100, 1),
                            'Medio': round((datos['Medio'] / total) * 100, 1),
                            'Alto': round((datos['Alto'] / total) * 100, 1)
                        },
                        'cantidades': {
                            'Bajo': datos['Bajo'],
                            'Medio': datos['Medio'],
                            'Alto': datos['Alto']
                        }
                    }
            
            # Identificar semestre más crítico
            semestre_critico = max(
                resultado_final.items(),
                key=lambda x: x[1]['porcentajes']['Alto']
            )
            
            if not resultado_final:
                return {
                    "tendencias_por_semestre": {},
                    "semestre_mas_critico": None,
                    "porcentaje_alto_critico": 0
                }
            
            return {
                'tendencias_por_semestre': resultado_final,
                'semestre_mas_critico': semestre_critico[0],
                'porcentaje_alto_critico': semestre_critico[1]['porcentajes']['Alto']
            }
            
        finally:
            cursor.close()
            conn.close()
    """
AGREGAR ESTE MÉTODO A analytics.py
===================================
Colócalo después de get_tendencia_por_semestre()
"""

    def get_evolucion_factor_por_semestre(self, factor: str) -> Dict:
        """
        Analiza la evolución de un factor específico por semestre
        
        Args:
            factor: Nombre de la variable (promedio_anterior, materias, trabajo, etc.)
        
        Returns:
            Evolución del factor por semestre con distribución de ansiedad
        """
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Mapeo de factores a condiciones SQL
            condiciones = {
                'promedio_anterior': 'c.promedio_anterior < 70',
                'materias': 'c.materias >= 7',
                'trabajo': 'c.trabajo = 1',
                'transporte': "c.transporte = 'Transporte publico'",
                'beca': 'c.beca = 1',
                'semestre': 'c.semestre <= 3',
                'edad': 'c.edad < 21',
                'familiares': 'c.familiares >= 5',
                'sexo': "c.sexo = 'F'",
                'estado_civil': "c.estado_civil != 'Soltero'",
                'carrera': "c.carrera = 'ISC'"  # Puedes hacer esto dinámico
            }
            
            condicion = condiciones.get(factor, 'c.promedio_anterior < 70')
            
            # Query para obtener distribución por semestre
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
            
            # Procesar resultados
            evolucion = {}
            for sem in range(1, 11):
                evolucion[sem] = {'Bajo': 0, 'Medio': 0, 'Alto': 0, 'total': 0}
            
            for row in resultados:
                sem = row['semestre']
                if sem in evolucion:
                    nivel = row['nivel']
                    cantidad = row['cantidad']
                    evolucion[sem][nivel] += cantidad
                    evolucion[sem]['total'] += cantidad
            
            # Convertir a formato final
            resultado = []
            for sem, datos in evolucion.items():
                total = datos['total']
                if total > 0:
                    resultado.append({
                        'semestre': sem,
                        'total': total,
                        'cantidades': {
                            'Bajo': datos['Bajo'],
                            'Medio': datos['Medio'],
                            'Alto': datos['Alto']
                        },
                        'porcentajes': {
                            'Bajo': round((datos['Bajo'] / total) * 100, 1),
                            'Medio': round((datos['Medio'] / total) * 100, 1),
                            'Alto': round((datos['Alto'] / total) * 100, 1)
                        }
                    })
            
            return {
                'factor': factor,
                'evolucion': resultado
            }
            
        finally:
            cursor.close()
            conn.close()

# Instancia global
analytics = AnxiTechAnalytics()
