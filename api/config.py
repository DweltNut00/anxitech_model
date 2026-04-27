"""
Configuracion de la API AnxiTech.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent.parent
SUPPORTED_DB_URL_SCHEMES = {
    "mariadb",
    "mariadb+pymysql",
    "mysql",
    "mysql+mysqlconnector",
    "mysql+pymysql",
}


def _load_local_dotenv() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    try:
        env_text = env_path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in env_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()

        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


_load_local_dotenv()


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si"}


def _parse_bool_value(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "si"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _get_int_value(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [
        item.strip().rstrip("/")
        for item in value.split(",")
        if item.strip()
    ]


def _resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate

    return candidate.resolve()


def _resolve_model_path() -> Path:
    configured_path = _resolve_path(_first_env("MODEL_PATH", "MODELO_PATH"))
    candidates = [
        configured_path,
        BASE_DIR / "modelos" / "modelo_ansiedad.pkl",
        BASE_DIR / "modelos" / "random_forest_ansiedad.pkl",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    return configured_path or candidates[1]


def _parse_database_url(raw_url: str | None) -> tuple[dict[str, object], dict[str, object]]:
    metadata: dict[str, object] = {
        "invalid": False,
        "present": bool(raw_url),
        "scheme": None,
        "supported": True,
    }
    if not raw_url:
        return {}, metadata

    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "").lower()
    metadata["scheme"] = scheme or None

    if scheme and scheme not in SUPPORTED_DB_URL_SCHEMES:
        metadata["supported"] = False
        return {}, metadata

    if not parsed.hostname:
        metadata["invalid"] = True
        return {}, metadata

    query_params = {
        key.lower().replace("-", "_"): values[-1]
        for key, values in parse_qs(parsed.query).items()
        if values
    }

    config: dict[str, object] = {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
    }

    if parsed.username:
        config["user"] = unquote(parsed.username)
    if parsed.password:
        config["password"] = unquote(parsed.password)

    database_name = unquote(parsed.path.lstrip("/")) if parsed.path else ""
    if database_name:
        config["database"] = database_name

    timeout_value = query_params.get("connection_timeout") or query_params.get("connect_timeout")
    if timeout_value:
        config["connection_timeout"] = _get_int_value(timeout_value, 30)

    ssl_disabled = _parse_bool_value(query_params.get("ssl_disabled"))
    if ssl_disabled is None:
        ssl_mode = (query_params.get("ssl_mode") or "").strip().lower()
        if ssl_mode:
            ssl_disabled = ssl_mode in {"0", "disabled", "false", "no", "off"}
        else:
            ssl_enabled = _parse_bool_value(query_params.get("ssl"))
            if ssl_enabled is not None:
                ssl_disabled = not ssl_enabled

    if ssl_disabled is not None:
        config["ssl_disabled"] = ssl_disabled

    for key in ("ssl_ca", "ssl_cert", "ssl_key"):
        value = query_params.get(key)
        if value:
            config[key] = value

    return config, metadata


DEFAULT_CORS_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
    "https://anxitechfrontend.netlify.app/",
    "https://railway.com/project/f9f67f8e-87ee-41bf-9a3a-09a6751751ae"
]

_cors_env = os.getenv("CORS_ORIGINS", "").strip()
CORS_ALLOW_ALL = _cors_env == "*"
CORS_ORIGINS = [] if CORS_ALLOW_ALL else _parse_csv_env("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
CORS_ALLOW_CREDENTIALS = _get_bool_env("CORS_ALLOW_CREDENTIALS", default=True)

API_TITLE = os.getenv("API_TITLE", "AnxiTech Analytics API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_DESCRIPTION = """
API de analisis para el sistema AnxiTech.

Proporciona estadisticas y analisis de datos de ansiedad estudiantil.

## Endpoints disponibles:

### Estadisticas
- `GET /api/stats/general` - Estadisticas generales del sistema
- `GET /api/stats/risk-factors` - Top 5 factores de riesgo
- `GET /api/stats/correlation` - Correlacion variable vs ansiedad
- `GET /api/stats/by-career` - Analisis por carrera
- `GET /api/stats/summary` - Resumen completo para dashboard

### Alertas
- `GET /api/stats/alerts` - Alertas tempranas

### Sistema
- `GET /api/health` - Estado de salud de la API
- `GET /api/modelo/info` - Informacion del modelo ML
"""

APP_HOST = _first_env("APP_HOST", "HOST", default="0.0.0.0")
APP_PORT = _get_int_value(_first_env("PORT", "APP_PORT"), 8000)
APP_RELOAD = _get_bool_env("APP_RELOAD", default=False)
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")
API_ROOT_PATH = (_first_env("API_ROOT_PATH", "ROOT_PATH", default="") or "").rstrip("/")

_DATABASE_URL_RAW = _first_env(
    "DATABASE_URL",
    "DB_URL",
    "MYSQL_URL",
    "MYSQL_PUBLIC_URL",
    "MYSQL_PRIVATE_URL",
    "CLEARDB_DATABASE_URL",
    "JAWSDB_URL",
)
_DATABASE_URL_CONFIG, _DATABASE_URL_METADATA = _parse_database_url(_DATABASE_URL_RAW)
_DB_TIMEOUT_DEFAULT = int(_DATABASE_URL_CONFIG.get("connection_timeout", 30))

DB_CONFIG = {
    "host": _first_env(
        "DB_HOST",
        "MYSQLHOST",
        "MYSQL_HOST",
        default=str(_DATABASE_URL_CONFIG.get("host", "localhost")),
    ),
    "port": _get_int_value(
        _first_env("DB_PORT", "MYSQLPORT", "MYSQL_PORT"),
        int(_DATABASE_URL_CONFIG.get("port", 3306)),
    ),
    "user": _first_env(
        "DB_USER",
        "MYSQLUSER",
        "MYSQL_USER",
        default=str(_DATABASE_URL_CONFIG.get("user", "root")),
    ),
    "password": _first_env(
        "DB_PASSWORD",
        "MYSQLPASSWORD",
        "MYSQL_PASSWORD",
        default=str(_DATABASE_URL_CONFIG.get("password", "")),
    ),
    "database": _first_env(
        "DB_NAME",
        "MYSQLDATABASE",
        "MYSQL_DATABASE",
        default=str(_DATABASE_URL_CONFIG.get("database", "anxitech")),
    ),
    "connection_timeout": _get_int_value(
        _first_env("DB_CONNECTION_TIMEOUT", "MYSQL_TIMEOUT"),
        _DB_TIMEOUT_DEFAULT,
    ),
}

_ssl_disabled = _parse_bool_value(_first_env("DB_SSL_DISABLED", "MYSQL_SSL_DISABLED"))
if _ssl_disabled is None:
    raw_ssl_disabled = _DATABASE_URL_CONFIG.get("ssl_disabled")
    _ssl_disabled = bool(raw_ssl_disabled) if isinstance(raw_ssl_disabled, bool) else None
if _ssl_disabled is not None:
    DB_CONFIG["ssl_disabled"] = _ssl_disabled

for config_key, env_names in {
    "ssl_ca": ("DB_SSL_CA", "MYSQL_SSL_CA"),
    "ssl_cert": ("DB_SSL_CERT", "MYSQL_SSL_CERT"),
    "ssl_key": ("DB_SSL_KEY", "MYSQL_SSL_KEY"),
}.items():
    env_value = _first_env(*env_names)
    fallback_value = _DATABASE_URL_CONFIG.get(config_key)
    if env_value:
        DB_CONFIG[config_key] = env_value
    elif isinstance(fallback_value, str) and fallback_value:
        DB_CONFIG[config_key] = fallback_value

for config_key, env_names in {
    "ssl_verify_cert": ("DB_SSL_VERIFY_CERT", "MYSQL_SSL_VERIFY_CERT"),
    "ssl_verify_identity": ("DB_SSL_VERIFY_IDENTITY", "MYSQL_SSL_VERIFY_IDENTITY"),
}.items():
    bool_value = _parse_bool_value(_first_env(*env_names))
    if bool_value is not None:
        DB_CONFIG[config_key] = bool_value

DB_CONFIG_PUBLIC: dict[str, object] = {
    "source": "database_url" if _DATABASE_URL_RAW else "env_vars",
    "host": DB_CONFIG["host"],
    "port": DB_CONFIG["port"],
    "database": DB_CONFIG["database"],
    "user": DB_CONFIG["user"],
    "connection_timeout": DB_CONFIG["connection_timeout"],
    "ssl_disabled": DB_CONFIG.get("ssl_disabled"),
    "ssl_ca_configured": "ssl_ca" in DB_CONFIG,
}

if _DATABASE_URL_METADATA.get("scheme"):
    DB_CONFIG_PUBLIC["url_scheme"] = _DATABASE_URL_METADATA["scheme"]
if _DATABASE_URL_RAW and not _DATABASE_URL_METADATA.get("supported", True):
    DB_CONFIG_PUBLIC["url_warning"] = (
        f"Unsupported database URL scheme: {_DATABASE_URL_METADATA.get('scheme')}"
    )
if _DATABASE_URL_RAW and _DATABASE_URL_METADATA.get("invalid", False):
    DB_CONFIG_PUBLIC["url_warning"] = "Database URL is invalid or missing a hostname"


def format_db_config_public() -> str:
    return ", ".join(
        f"{key}={value}"
        for key, value in DB_CONFIG_PUBLIC.items()
        if value not in (None, "")
    )


MODELO_PATH = _resolve_model_path()
