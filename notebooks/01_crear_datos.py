# 01_crear_datos.py - Crear datos de prueba CON OPCIONES REALES
import pandas as pd
import numpy as np
import os

print("=== CREANDO DATOS DE PRUEBA CON OPCIONES REALES ===")

# Configurar semilla para reproducibilidad
np.random.seed(42)
n_estudiantes = 150  # 150 estudiantes

# 1. DATOS ACADÉMICOS
print("Creando datos académicos...")
grades = pd.DataFrame({
    'uid': [f'estudiante_{i:03d}' for i in range(1, n_estudiantes+1)],
    'promedio': np.round(np.random.normal(8.0, 1.0, n_estudiantes), 1), 
    'semestre': np.random.randint(1, 11, n_estudiantes),  # Semestre 1-10
    'num_materias': np.random.randint(3, 9, n_estudiantes)  # 3-8 materias
})

# Asegurar promedio en rango válido (6-10)
grades['promedio'] = np.clip(grades['promedio'], 6.0, 10.0)

# 2. DATOS SOCIODEMOGRÁFICOS 
print("Creando datos sociodemográficos...")

# OPCIONES REALES DE LA APLICACIÓN
carreras = [
    'En sistemas computacionales',
    'Informatica',
    'Ciencia de datos',
    'Gestion empresarial',
    'Química',
    'Electrica',
    'Electronica',
    'En semiconductores',
    'Industrial',
    'Industrial (En linea)',
    'Mecanica'
]

transportes = [
    'Transporte publico',
    'Vehiculo particular', 
    'Caminando',
    'No me transporto (En linea)'
]

hogares = [
    'Vivo solo',
    'Con familiares',
    'Con amigos', 
    'Con conocidos'
]

# Crear DataFrame 
socio = pd.DataFrame({
    'uid': [f'estudiante_{i:03d}' for i in range(1, n_estudiantes+1)],
    'carrera': np.random.choice(carreras, n_estudiantes),
    'transporte': np.random.choice(transportes, n_estudiantes, 
                                 p=[0.45, 0.30, 0.20, 0.05]),  # Distribución realista
    'hogar': np.random.choice(hogares, n_estudiantes,
                             p=[0.15, 0.70, 0.10, 0.05]),     # Mayoría con familia
    'trabaja': np.random.choice([0, 1], n_estudiantes, p=[0.65, 0.35]),  # 35% trabaja
    'beca': np.random.choice([0, 1], n_estudiantes, p=[0.70, 0.30])      # 30% tiene beca
})

# 3. DATOS DE ANSIEDAD (SIMULANDO DASS-21 CON CORRELACIONES REALISTAS)
print("Creando datos de ansiedad...")

# Crear correlaciones realistas específicas:
# - Más materias = más ansiedad
# - Trabajar = más ansiedad  
# - Mejor promedio = menos ansiedad
# - Carreras técnicas (sistemas, electrónica) = más ansiedad
# - Transporte público = más ansiedad (estrés del viaje)
# - Vivir solo = más ansiedad

# Factor por carrera (algunas ingenierías son más estresantes)
factor_carrera = []
for carrera in socio['carrera']:
    if carrera in ['En sistemas computacionales', 'Electronica', 'En semiconductores', 'Ciencia de datos']:
        factor_carrera.append(3)  # Más estresantes (tecnología avanzada)
    elif carrera in ['Industrial', 'Mecanica', 'Electrica']:
        factor_carrera.append(2)  # Moderadamente estresantes
    elif carrera in ['Informatica', 'Industrial (En linea)']:
        factor_carrera.append(1)  # Menos estresantes
    else:  # Gestión empresarial, Química
        factor_carrera.append(1.5)  # Moderadamente estresantes
factor_carrera = np.array(factor_carrera)

# Factor por transporte
factor_transporte = []
for transporte in socio['transporte']:
    if transporte == 'Transporte publico':
        factor_transporte.append(2)  # Más estresante (tiempo, multitudes)
    elif transporte == 'Caminando':
        factor_transporte.append(-1)  # Menos estresante (ejercicio, relajante)
    elif transporte == 'No me transporto (En linea)':
        factor_transporte.append(0)  # Neutral (comodidad vs aislamiento)
    else:  # Vehículo particular
        factor_transporte.append(1)  # Ligeramente estresante (tráfico, gasolina)
factor_transporte = np.array(factor_transporte)

# Factor por hogar
factor_hogar = []
for hogar in socio['hogar']:
    if hogar == 'Vivo solo':
        factor_hogar.append(3)  # Más ansiedad (soledad, responsabilidades)
    elif hogar == 'Con familiares':
        factor_hogar.append(-1)  # Menos ansiedad (apoyo familiar)
    elif hogar == 'Con amigos':
        factor_hogar.append(1)  # Neutral-positivo
    else:  # Con conocidos
        factor_hogar.append(2)  # Algo estresante (tensión social)
factor_hogar = np.array(factor_hogar)

# Calcular ansiedad con correlaciones realistas
ansiedad_base = (
    grades['num_materias'] * 2.5 +                    # Materias influyen fuertemente
    socio['trabaja'] * 4 +                           # Trabajar aumenta ansiedad
    (10 - grades['promedio']) * 2 +                  # Menor promedio = más ansiedad
    factor_carrera +                                 # Factor por carrera
    factor_transporte +                              # Factor por transporte  
    factor_hogar +                                   # Factor por hogar
    (grades['semestre'] > 8) * 2 +                   # Semestres avanzados más estresantes
    np.random.normal(0, 2.5, n_estudiantes)         # Ruido aleatorio
)

# Convertir a escala DASS-21 (0-21)
ansiedad_score = np.clip(ansiedad_base, 0, 21)

# Crear niveles categóricos según DASS-21 real
def score_to_level(score):
    if score <= 7:
        return "Bajo"
    elif score <= 14:
        return "Medio"  
    else:
        return "Alto"

ansiedad = pd.DataFrame({
    'uid': [f'estudiante_{i:03d}' for i in range(1, n_estudiantes+1)],
    'dass21_score': np.round(ansiedad_score, 1),
    'nivel_ansiedad': [score_to_level(s) for s in ansiedad_score]
})

# 4. UNIR TODO EN DATASET FINAL
print("Uniendo datos...")
dataset = grades.merge(socio, on='uid')
dataset = dataset.merge(ansiedad, on='uid')

# 5. GUARDAR ARCHIVOS
print("💾 Guardando archivos...")
os.makedirs('datos', exist_ok=True)

dataset.to_csv('datos/dataset_completo.csv', index=False)
grades.to_csv('datos/academicos.csv', index=False)
socio.to_csv('datos/sociodemograficos.csv', index=False)
ansiedad.to_csv('datos/ansiedad.csv', index=False)

print("✅ Archivos creados:")
print("   - datos/dataset_completo.csv")
print("   - datos/academicos.csv")
print("   - datos/sociodemograficos.csv")
print("   - datos/ansiedad.csv")

# 6. MOSTRAR ESTADÍSTICAS
print(f"\n RESUMEN DEL DATASET:")
print(f"Total estudiantes: {len(dataset)}")
print(f"Columnas: {list(dataset.columns)}")

print(f"\nDistribución de ansiedad:")
print(dataset['nivel_ansiedad'].value_counts())

print(f"\nDistribución por carrera (top 5):")
print(dataset['carrera'].value_counts().head())

print(f"\nDistribución de transporte:")
print(dataset['transporte'].value_counts())

print(f"\nDistribución de hogar:")
print(dataset['hogar'].value_counts())

print(f"\nPrimeros 5 registros:")
print(dataset.head())

print(f"\nEstadísticas de ansiedad por carrera:")
ansiedad_por_carrera = dataset.groupby('carrera')['dass21_score'].agg(['mean', 'count']).round(2)
print(ansiedad_por_carrera.sort_values('mean', ascending=False))

print("\n ¡Datos creados exitosamente!")