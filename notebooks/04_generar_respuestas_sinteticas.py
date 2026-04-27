import mysql.connector
import random
from datetime import datetime, timedelta

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="anxitech"
)
cursor = conn.cursor()

print("="*70)
print("GENERAR DATOS DASS-21 CON 500 ALUMNOS SINTÉTICOS")
print("="*70)

# Aplicar recategorización primero
print("\n🔧 Aplicando recategorización...")

try:
    # ESTRÉS
    cursor.execute("UPDATE pregunta SET categoria = 'estres' WHERE id IN (2,3,7,12,13,15,19)")
    
    # DEPRESIÓN
    cursor.execute("UPDATE pregunta SET categoria = 'depresion' WHERE id IN (4,6,11,14,17,18,23)")
    
    # ANSIEDAD
    cursor.execute("UPDATE pregunta SET categoria = 'ansiedad' WHERE id IN (5,8,10,16,21,22,25)")
    
    # DESACTIVAR duplicadas
    cursor.execute("UPDATE pregunta SET status = 0 WHERE id IN (20,24)")
    
    conn.commit()
    print("✅ Categorías actualizadas\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

# Verificar
cursor.execute("""
    SELECT categoria, COUNT(*) as cant
    FROM pregunta 
    WHERE status = 1
    GROUP BY categoria
""")

print("📊 Preguntas por categoría:")
for cat, cant in cursor.fetchall():
    print(f"   {cat}: {cant}")

# Obtener solo preguntas de ANSIEDAD
cursor.execute("""
    SELECT id FROM pregunta 
    WHERE status = 1 AND categoria = 'ansiedad'
    ORDER BY id
""")
preguntas_ansiedad = [row[0] for row in cursor.fetchall()]

if len(preguntas_ansiedad) != 7:
    print(f"\n❌ ERROR: Se esperaban 7 preguntas de ansiedad, hay {len(preguntas_ansiedad)}")
    exit()

print(f"\n✅ Preguntas de ansiedad: {preguntas_ansiedad}\n")

# Aplicación
cursor.execute("SELECT id FROM aplicacion WHERE tipo = 0 AND status = 1 LIMIT 1")
result = cursor.fetchone()
if result:
    id_aplicacion = result[0]
else:
    inicio = datetime.now().strftime('%Y-%m-%d')
    fin = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    cursor.execute("INSERT INTO aplicacion (inicio, fin, tipo, status) VALUES (%s, %s, 0, 1)", (inicio, fin))
    id_aplicacion = cursor.lastrowid
    conn.commit()

# ============================================
# PATRONES DASS-21 ANSIEDAD (7 preguntas, 0-21 puntos)
# ============================================

patrones = {
    'bajo': {
        'suma_objetivo': 3,    # Promedio: 0.43 por pregunta
        'rango': (0, 7),
        'descripcion': 'Normal - Sin ansiedad significativa'
    },
    'medio': {
        'suma_objetivo': 11,   # Promedio: 1.57 por pregunta
        'rango': (8, 14),
        'descripcion': 'Leve a Moderado - Ansiedad presente'
    },
    'alto': {
        'suma_objetivo': 17,   # Promedio: 2.43 por pregunta
        'rango': (15, 21),
        'descripcion': 'Severo a Extremo - Ansiedad clínica'
    }
}

print("📊 Rangos DASS-21 Ansiedad:")
for nivel, config in patrones.items():
    print(f"   {nivel.upper()}: {config['rango'][0]}-{config['rango'][1]} pts - {config['descripcion']}")

# Distribución: 40% bajo, 40% medio, 20% alto
distribucion = ['bajo'] * 200 + ['medio'] * 200 + ['alto'] * 100  # Total: 500
random.shuffle(distribucion)

# Configuración
NUM_ALUMNOS = 500  # ⭐ CAMBIADO A 500
carreras = ['ISC', 'IINF', 'ICD', 'IGE']
sexos = ['M', 'F']
estados_civiles = ['Soltero', 'Casado']

print(f"\n👥 Generando {NUM_ALUMNOS} alumnos...\n")

contador = 0
stats = {'bajo': 0, 'medio': 0, 'alto': 0}

for i in range(NUM_ALUMNOS):
    try:
        # Usuario
        nombre = f"Alumno{i+1}"
        apellido = f"Test{i+1}"
        timestamp = datetime.now().strftime('%H%M%S%f')
        usuario = f"alumno{i+1}_{timestamp}"
        email = f"alumno{i+1}_{random.randint(10000,99999)}@test.com"
        password = "$2y$10$abcdefgh"
        
        cursor.execute("""
            INSERT INTO usuario (usuario, nombre, apellido, email, password, tema, status, creacion)
            VALUES (%s, %s, %s, %s, %s, 'light', 1, NOW())
        """, (usuario, nombre, apellido, email, password))
        
        id_usuario = cursor.lastrowid
        
        # Alumno
        nocontrol = f"{2020 + (i // 100)}{10000 + i}"
        edad = random.randint(18, 28)
        fecha_nac = (datetime.now() - timedelta(days=365*edad)).strftime('%Y-%m-%d')
        sexo = random.choices(sexos, weights=[55, 45])[0]
        estado_civil = random.choices(estados_civiles, weights=[80, 20])[0]
        
        cursor.execute("""
            INSERT INTO alumno (id, nocontrol, fechan, sexo, estadoc, ciudad, estado)
            VALUES (%s, %s, %s, %s, %s, 'Orizaba', 'Veracruz')
        """, (id_usuario, nocontrol, fecha_nac, sexo, estado_civil))
        
        # Complemento
        carrera = random.choice(carreras)
        semestre = random.randint(1, 10)
        promedio = round(random.uniform(70, 95), 1)
        materias = random.randint(4, 8)
        transporte = random.randint(0, 3)
        trabajo = random.choices([0, 1], weights=[70, 30])[0]
        beca = random.choices([0, 1], weights=[75, 25])[0]
        
        cursor.execute("""
            INSERT INTO complemento 
            (id_alumno, id_aplicacion, carrera, promedio_anterior, semestre, materias, 
             transporte, familiares, trabajo, beca, sexo, edad, estado_civil)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (id_usuario, id_aplicacion, carrera, promedio, semestre, materias,
              transporte, random.randint(0,3), trabajo, beca, sexo, edad, estado_civil))
        
        # ============================================
        # GENERAR RESPUESTAS DE ANSIEDAD (7 preguntas)
        # ============================================
        nivel = distribucion[i % len(distribucion)]
        suma_objetivo = patrones[nivel]['suma_objetivo']
        rango_min, rango_max = patrones[nivel]['rango']
        
        # Generar 7 respuestas que sumen dentro del rango
        respuestas = []
        suma_actual = 0
        
        for j in range(7):
            restantes = 7 - j
            falta = suma_objetivo - suma_actual
            promedio_necesario = falta / restantes if restantes > 0 else 0
            
            # Generar valor con variabilidad
            if nivel == 'bajo':
                valor = max(0, min(2, int(promedio_necesario + random.uniform(-0.4, 0.6))))
            elif nivel == 'medio':
                valor = max(0, min(3, int(promedio_necesario + random.uniform(-0.6, 0.6))))
            else:  # alto
                valor = max(1, min(3, int(promedio_necesario + random.uniform(-0.4, 0.6))))
            
            respuestas.append(valor)
            suma_actual += valor
        
        # Ajustar para que quede dentro del rango
        suma_final = sum(respuestas)
        
        while suma_final < rango_min:
            idx = random.randint(0, 6)
            if respuestas[idx] < 3:
                respuestas[idx] += 1
                suma_final += 1
        
        while suma_final > rango_max:
            idx = random.randint(0, 6)
            if respuestas[idx] > 0:
                respuestas[idx] -= 1
                suma_final -= 1
        
        # Insertar respuestas
        for pregunta_id, valor in zip(preguntas_ansiedad, respuestas):
            cursor.execute("""
                INSERT INTO alumno_pregunta (id_alumno, id_pregunta, valor, id_aplicacion)
                VALUES (%s, %s, %s, %s)
            """, (id_usuario, pregunta_id, valor, id_aplicacion))
        
        # Clasificar según suma
        suma_final = sum(respuestas)
        if suma_final <= 7:
            nivel_real = 'bajo'
        elif suma_final <= 14:
            nivel_real = 'medio'
        else:
            nivel_real = 'alto'
        
        emoji = {'bajo': '🟢', 'medio': '🟡', 'alto': '🔴'}
        
        # Mostrar progreso cada 25 alumnos
        if i % 25 == 0:
            print(f"{emoji[nivel_real]} {nombre} - {carrera}, Ansiedad: {suma_final}/21, {nivel_real.upper()}")
        
        contador += 1
        stats[nivel_real] += 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        continue

conn.commit()

# Estadísticas
print("\n" + "="*70)
print("GENERACIÓN COMPLETADA - 500 ALUMNOS")
print("="*70)
print(f"\nAlumnos creados: {contador}")
print(f"\n Distribución DASS-21:")
total = sum(stats.values())
for nivel, count in stats.items():
    print(f"   {emoji[nivel]} {nivel.upper():<8}: {count:3d} ({count/total*100:.1f}%)")

cursor.execute("SELECT AVG(valor), SUM(valor) FROM alumno_pregunta")
avg, suma = cursor.fetchone()
print(f"\nPromedio por pregunta: {avg:.2f}/3")
print(f"Suma promedio por alumno: {suma/contador:.1f}/21")

cursor.close()
conn.close()

print("\n Próximos pasos:")
print("   1. python 05_analisis_dataset_corregido.py")
print("   2. python 06_entrenar_modelo.py")
print("="*70)