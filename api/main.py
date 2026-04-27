"""
API Principal de AnxiTech.
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if __package__:
    from .analytics import analytics
    from .config import (
        API_DESCRIPTION,
        API_ROOT_PATH,
        API_TITLE,
        API_VERSION,
        APP_HOST,
        APP_LOG_LEVEL,
        APP_PORT,
        APP_RELOAD,
        CORS_ALLOW_ALL,
        CORS_ALLOW_CREDENTIALS,
        CORS_ORIGINS,
        format_db_config_public,
    )
    from .models import (
        Alertas,
        AnalisisCorrelacion,
        CarrerasResponse,
        EstadisticasGenerales,
        FactoresRiesgoResponse,
        HealthResponse,
    )
else:
    from analytics import analytics
    from config import (
        API_DESCRIPTION,
        API_ROOT_PATH,
        API_TITLE,
        API_VERSION,
        APP_HOST,
        APP_LOG_LEVEL,
        APP_PORT,
        APP_RELOAD,
        CORS_ALLOW_ALL,
        CORS_ALLOW_CREDENTIALS,
        CORS_ORIGINS,
        format_db_config_public,
    )
    from models import (
        Alertas,
        AnalisisCorrelacion,
        CarrerasResponse,
        EstadisticasGenerales,
        FactoresRiesgoResponse,
        HealthResponse,
    )


VALID_CORRELATION_FIELDS = [
    "promedio_anterior",
    "semestre",
    "materias",
    "edad",
    "transporte",
    "familiares",
    "trabajo",
    "beca",
    "sexo",
    "estado_civil",
    "carrera",
]

VALID_EVOLUTION_FACTORS = [
    "promedio_anterior",
    "materias",
    "trabajo",
    "transporte",
    "beca",
    "semestre",
    "edad",
    "familiares",
    "sexo",
    "estado_civil",
    "carrera",
]


def _docs_path() -> str:
    return f"{API_ROOT_PATH}/docs" if API_ROOT_PATH else "/docs"


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    root_path=API_ROOT_PATH,
)

cors_config = {
    "allow_credentials": CORS_ALLOW_CREDENTIALS,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

if CORS_ALLOW_ALL:
    if CORS_ALLOW_CREDENTIALS:
        cors_config["allow_origin_regex"] = ".*"
    else:
        cors_config["allow_origins"] = ["*"]
else:
    cors_config["allow_origins"] = CORS_ORIGINS

app.add_middleware(CORSMiddleware, **cors_config)


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz."""
    return {
        "mensaje": "AnxiTech Analytics API",
        "version": API_VERSION,
        "documentacion": _docs_path(),
        "estado": "activo",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica el estado de salud de la API."""
    try:
        conn = analytics.get_db_connection()
        db_conectada = conn.is_connected()
        conn.close()
    except Exception:
        db_conectada = False

    return {
        "status": "ok" if db_conectada and analytics.modelo else "degraded",
        "modelo_cargado": analytics.modelo is not None,
        "db_conectada": db_conectada,
        "version": API_VERSION,
    }


@app.get("/api/stats/general", response_model=EstadisticasGenerales, tags=["Estadisticas"])
async def estadisticas_generales():
    """Obtiene estadisticas generales del sistema."""
    try:
        return analytics.get_estadisticas_generales()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadisticas: {e}") from e


@app.get("/api/stats/risk-factors", response_model=FactoresRiesgoResponse, tags=["Analisis"])
async def factores_riesgo():
    """Obtiene el top 5 de factores de riesgo."""
    try:
        factores = analytics.get_factores_riesgo()
        if not factores:
            raise HTTPException(status_code=503, detail="Modelo ML no disponible")

        return {
            "factores": factores,
            "fecha_analisis": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener factores de riesgo: {e}") from e


@app.get("/api/stats/correlation", response_model=AnalisisCorrelacion, tags=["Analisis"])
async def correlacion_variable(variable: str, umbral: float):
    """Analiza la correlacion entre una variable y el nivel de ansiedad."""
    try:
        if variable not in VALID_CORRELATION_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Variable invalida. Validas: {', '.join(VALID_CORRELATION_FIELDS)}",
            )

        return analytics.get_correlacion_variable(variable, umbral)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar correlacion: {e}") from e


@app.get("/api/stats/by-career", response_model=CarrerasResponse, tags=["Analisis"])
async def analisis_por_carrera():
    """Analiza los niveles de ansiedad por carrera."""
    try:
        return analytics.get_analisis_por_carrera()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar por carrera: {e}") from e


@app.get("/api/stats/alerts", response_model=Alertas, tags=["Alertas"])
async def alertas_tempranas():
    """Obtiene alertas tempranas del sistema."""
    try:
        return analytics.get_alertas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener alertas: {e}") from e


@app.get("/api/stats/summary", tags=["Dashboard"])
async def resumen_dashboard():
    """Combina multiples estadisticas en una sola llamada."""
    try:
        return {
            "estadisticas_generales": analytics.get_estadisticas_generales(),
            "factores_riesgo": analytics.get_factores_riesgo(),
            "analisis_carreras": analytics.get_analisis_por_carrera(),
            "alertas": analytics.get_alertas(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen: {e}") from e


@app.get("/api/modelo/info", tags=["Modelo ML"])
async def info_modelo():
    """Informacion sobre el modelo ML cargado."""
    if not analytics.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return {
        "modelo_tipo": "Random Forest Classifier",
        "features": analytics.feature_names,
        "num_features": len(analytics.feature_names),
        "modelo_cargado": True,
        "n_estimators": analytics.modelo.n_estimators if hasattr(analytics.modelo, "n_estimators") else None,
    }


@app.get("/api/stats/test-db", tags=["Testing"])
async def test_database():
    """Endpoint para verificar conexion a base de datos."""
    try:
        conn = analytics.get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM usuario")
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return {
            "status": "ok",
            "mensaje": "Conexion exitosa a la base de datos",
            "total_usuarios": result[0],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de conexion: {e}") from e


@app.get("/api/alumno/{id_alumno}/explicacion", tags=["Analisis Individual"])
async def explicacion_nivel_ansiedad(id_alumno: int):
    """Explica el patron de decision para un alumno."""
    try:
        return analytics.get_explicacion_individual(id_alumno)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/stats/correlaciones", tags=["Analisis"])
async def tabla_correlaciones():
    """Tabla completa de correlaciones."""
    try:
        correlaciones = analytics.get_tabla_correlaciones()
        return {
            "correlaciones": correlaciones,
            "total": len(correlaciones),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e


@app.get("/api/stats/tendencias-semestre", tags=["Analisis"])
def get_tendencias_semestre():
    """Obtiene evolucion de ansiedad por semestre."""
    try:
        return analytics.get_tendencia_por_semestre()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/stats/evolucion-factor", tags=["Analisis"])
async def evolucion_factor_por_semestre(factor: str):
    """Obtiene la evolucion de un factor especifico por semestre."""
    try:
        if factor not in VALID_EVOLUTION_FACTORS:
            raise HTTPException(
                status_code=400,
                detail=f"Factor invalido. Validos: {', '.join(VALID_EVOLUTION_FACTORS)}",
            )

        return analytics.get_evolucion_factor_por_semestre(factor)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint no encontrado",
            "mensaje": f"Verifica la ruta en {_docs_path()}",
            "path": str(request.url),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "mensaje": "Contacta al administrador si el problema persiste",
        },
    )


@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la API."""
    print("=" * 70)
    print("INICIANDO ANXITECH ANALYTICS API")
    print("=" * 70)
    print(f"API Version: {API_VERSION}")
    print(f"Modelo ML: {'Cargado' if analytics.modelo else 'No disponible'}")
    print(f"Base de datos objetivo: {format_db_config_public()}")

    try:
        conn = analytics.get_db_connection()
        print("Base de datos: Conectada")
        conn.close()
    except Exception:
        print("Base de datos: Error de conexion")

    print("=" * 70)
    print(f"Documentacion disponible en: {_docs_path()}")
    print("=" * 70)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_RELOAD,
        log_level=APP_LOG_LEVEL,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
