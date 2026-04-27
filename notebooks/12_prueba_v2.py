import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar CSV
csv_files = list((BASE_DIR / "analisis_dataset").glob("dataset_completo_*.csv"))
df = pd.read_csv(max(csv_files, key=lambda p: p.stat().st_mtime))

# Cargar modelo
modelo = joblib.load(BASE_DIR / "modelos" / "modelo_ansiedad.pkl")

# Preparar features (igual que en entrenamiento)
feature_names = ['promedio_anterior', 'semestre', 'materias', 'edad',
                 'transporte', 'familiares', 'trabajo', 'beca',
                 'sexo', 'estado_civil', 'carrera']

X = df[feature_names].copy()

# Encodear categóricas
for col in ['sexo', 'estado_civil', 'carrera']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Predecir
df['nivel_predicho'] = modelo.predict(X)

# Perfil por nivel PREDICHO
perfil = df.groupby('nivel_predicho').agg(
    n            = ('nivel_predicho', 'count'),
    promedio     = ('promedio_anterior', 'mean'),
    materias     = ('materias', 'mean'),
    semestre     = ('semestre', 'mean'),
    pct_trabaja  = ('trabajo', 'mean'),
    transporte   = ('transporte', 'mean'),
    familiares   = ('familiares', 'mean'),
    pct_femenino = ('sexo', lambda x: (x == 'F').mean() * 100)
).round(1)

print(perfil.to_string())
print("\nDistribución:")
print(df['nivel_predicho'].value_counts())
