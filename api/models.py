"""
Modelos de datos para la API
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel

# Modelo para estadísticas generales
class EstadisticasGenerales(BaseModel):
    total_alumnos: int
    distribucion: Dict[str, int]
    porcentajes: Dict[str, float]
    ansiedad_promedio: float

# Modelo para un factor de riesgo
class FactorRiesgo(BaseModel):
    rank: int
    nombre: str
    variable: str
    importancia: float
    porcentaje_impacto: float
    descripcion: str

# Modelo para respuesta de factores de riesgo
class FactoresRiesgoResponse(BaseModel):
    factores: List[FactorRiesgo]
    fecha_analisis: str

# Modelo para correlación
class CorrelacionDistribucion(BaseModel):
    cantidad: int
    porcentaje: float

class AnalisisCorrelacion(BaseModel):
    variable: str
    condicion: str
    total_alumnos_condicion: int
    distribucion_ansiedad: Dict[str, CorrelacionDistribucion]

# Modelo para análisis por carrera
class CarreraAnalisis(BaseModel):
    nombre: str
    total_alumnos: int
    ansiedad_promedio: float
    distribucion: Dict[str, int]
    porcentajes: Dict[str, float]
    nivel_riesgo: str

class CarrerasResponse(BaseModel):
    carreras: List[CarreraAnalisis]
    carrera_mayor_riesgo: str

# Modelo para alertas
class Alertas(BaseModel):
    alumnos_riesgo_alto: int
    porcentaje_riesgo_alto: float
    carrera_mayor_riesgo: Dict[str, Any]  # ← AQUÍ ESTABA EL ERROR
    semestre_critico: Optional[Dict[str, Any]]  # ← Y AQUÍ
    factor_principal: str

# Modelo para respuesta de salud de la API
class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    db_conectada: bool
    version: str