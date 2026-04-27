import mysql.connector
import random
from datetime import datetime, timedelta

print("="*70)
print("SETUP COMPLETO - GENERAR DATOS SINTÉTICOS")
print("="*70)

# ============================================
# 1. CONEXIÓN
# ============================================
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="anxitech"
    )
    cursor = conn.cursor()
    print("✅ Conexión exitosa\n")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    exit()

# ============================================
# 2. VERIFICAR ESTRUCTURA
# ============================================
print("📋 Verificando estructura de base de datos...\n")

tablas_requeridas = ['usuario', 'alumno', 'admin', 'complemento', 'alumno_pregunta', 'pregunta', 'aplicacion']
tablas_existentes = []

for tabla in tablas_requeridas:
    cursor.execute(f"SHOW TABLES LIKE '{tabla}'")
    if cursor.fetchone():
        tablas_existentes.append(tabla)
        print(f"  ✅ Tabla '{tabla}' existe")
    else:
        print(f"  ❌ Tabla '{tabla}' NO existe")

if len(tablas_existentes) != len(tablas_requeridas):
    print("\n❌ ERROR: Faltan tablas. Restaura el backup SQL primero.")
    cursor.close()
    conn.close()
    exit()

# ============================================
# 3. VERIFICAR DATOS EXISTENTES
# ============================================
print("\n" + "="*70)
print("📊 DATOS ACTUALES")
print("="*70)

cursor.execute("SELECT COUNT(*) FROM usuario")
total_usuarios = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM complemento")
total_complementos = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM alumno_pregunta")
total_respuestas = cursor.fetchone()[0]

print(f"Usuarios: {total_usuarios}")
print(f"Cuestionarios complementarios: {total_complementos}")
print(f"Respuestas de ansiedad: {total_respuestas}")

if total_usuarios > 0:
    print("\n⚠️  Ya hay datos en la base de datos.")
    print("¿Deseas ELIMINAR TODO y regenerar desde cero? (s/n): ", end='')
    respuesta = input().strip().lower()
    
    if respuesta != 's':
        print("👋 Operación cancelada")
        cursor.close()
        conn.close()
        exit()
    
    # Limpiar tablas
    print("\n🗑️  Limpiando datos existentes...")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("DELETE FROM alumno_pregunta")
    cursor.execute("DELETE FROM complemento")
    cursor.execute("DELETE FROM alumno WHERE id NOT IN (SELECT id FROM admin)")
    cursor.execute("DELETE FROM usuario WHERE id NOT IN (SELECT id FROM admin)")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    print("✅ Datos limpiados\n")

# ============================================
# 4. VERIFICAR PREGUNTAS
# ============================================
cursor.execute("SELECT COUNT(*) FROM pregunta WHERE status = 1")
preguntas_activas = cursor.fetchone()[0]

if preguntas_activas < 15:
    print(f"❌ ERROR: Solo hay {preguntas_activas} preguntas activas")
    print("   Se necesitan al menos 15 preguntas del test DAS-42")
    cursor.close()
    conn.close()
    exit()

cursor.execute("SELECT id FROM pregunta WHERE status = 1 ORDER BY id LIMIT 15")
preguntas_ids = [row[0] for row in cursor.fetchall()]
print(f"✅ Preguntas disponibles: {len(preguntas_ids)}")

# ============================================
# 5. VERIFICAR/CREAR PERÍODO DE APLICACIÓN
# ============================================
cursor.execute("SELECT id FROM aplicacion WHERE tipo = 0 AND status = 1 LIMIT 1")
result = cursor.fetchone()

if result:
    id_aplicacion = result[0]
    print(f"✅ Período de aplicación existente: {id_aplicacion}")
else:
    # Crear período de aplicación
    inicio = datetime.now().strftime('%Y-%m-%d')
    fin = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    cursor.execute("""
        INSERT INTO aplicacion (inicio, fin, tipo, status) 
        VALUES (%s, %s, 0, 1)
    """, (inicio, fin))
    id_aplicacion = cursor.lastrowid
    conn.commit()
    print(f"✅ Período de aplicación creado: {id_aplicacion}")

# ============================================
# 6. GENERAR ALUMNOS SINTÉTICOS
# ============================================
print("\n" + "="*70)
print("👥 GENERANDO ALUMNOS SINTÉTICOS")
print("="*70)

NUM_ALUMNOS = 20  # Puedes cambiar este número

carreras = ['ISC', 'IINF', 'ICD', 'IGE']
sexos = ['M', 'F']
estados_civiles = ['Soltero', 'Casado', 'Divorciado']
ciudades = ['Orizaba', 'Córdoba', 'Fortín']

alumnos_creados = []

for i in range(NUM_ALUMNOS):
    try:
        # Crear usuario
        nombre = f"Alumno{i+1}"
        apellido = f"Test{i+1}"
        usuario = f"alumno{i+1}"
        email = f"alumno{i+1}@test.com"
        password_hash = "$2y$10$abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGH"
        
        cursor.execute("""
            INSERT INTO usuario (usuario, nombre, apellido, email, password, tema, status, creacion)
            VALUES (%s, %s, %s, %s, %s, 'light', 1, NOW())
        """, (usuario, nombre, apellido, email, password_hash))
        
        id_usuario = cursor.lastrowid
        
        # Crear alumno
        nocontrol = f"200{10000 + i}"
        edad = random.randint(18, 25)
        fecha_nacimiento = (datetime.now() - timedelta(days=365*edad)).strftime('%Y-%m-%d')
        sexo = random.choice(sexos)
        estado_civil = random.choice(estados_civiles)
        ciudad = random.choice(ciudades)
        
        cursor.execute("""
            INSERT INTO alumno (id, nocontrol, fechan, sexo, estadoc, ciudad, estado)
            VALUES (%s, %s, %s, %s, %s, %s, 'Veracruz')
        """, (id_usuario, nocontrol, fecha_nacimiento, sexo, estado_civil, ciudad))
        
        alumnos_creados.append({
            'id': id_usuario,
            'nombre': nombre,
            'apellido': apellido,
            'edad': edad,
            'sexo': sexo,
            'estado_civil': estado_civil
        })
        
        print(f"  ✅ {nombre} {apellido} - {sexo}, {edad} años, {estado_civil}")
    
    except Exception as e:
        print(f"  ❌ Error creando alumno {i+1}: {e}")
        conn.rollback()
        continue

conn.commit()
print(f"\n✅ {len(alumnos_creados)} alumnos creados\n")

# ============================================
# 7. GENERAR CUESTIONARIOS COMPLEMENTARIOS
# ============================================
print("="*70)
print("📝 GENERANDO CUESTIONARIOS COMPLEMENTARIOS")
print("="*70)

for alumno in alumnos_creados:
    try:
        carrera = random.choice(carreras)
        promedio = round(random.uniform(70, 100), 1)
        semestre = random.randint(1, 10)
        materias = random.randint(4, 8)
        transporte = random.randint(0, 3)
        familiares = random.randint(0, 3)
        trabajo = random.randint(0, 1)
        beca = random.randint(0, 1)
        
        cursor.execute("""
            INSERT INTO complemento 
            (id_alumno, id_aplicacion, carrera, promedio_anterior, semestre, materias, 
             transporte, familiares, trabajo, beca, sexo, edad, estado_civil)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (alumno['id'], id_aplicacion, carrera, promedio, semestre, materias,
              transporte, familiares, trabajo, beca, alumno['sexo'], alumno['edad'], alumno['estado_civil']))
        
        print(f"  ✅ {alumno['nombre']} - {carrera}, Sem {semestre}, Prom {promedio}")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        conn.rollback()

conn.commit()

# ============================================
# 8. GENERAR RESPUESTAS DE ANSIEDAD
# ============================================
print("\n" + "="*70)
print("🧠 GENERANDO RESPUESTAS DE ANSIEDAD (3 NIVELES)")
print("="*70)

patrones = {
    'bajo':  [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    'medio': [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2],
    'alto':  [2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3]
}

distribucion = ['bajo']*8 + ['medio']*8 + ['alto']*4
random.shuffle(distribucion)

for idx, alumno in enumerate(alumnos_creados):
    nivel = distribucion[idx % len(distribucion)]
    patron_base = patrones[nivel].copy()
    
    # Variabilidad
    respuestas = []
    for valor in patron_base:
        variacion = random.choice([-1, 0, 0, 1])
        valor_final = max(0, min(4, valor + variacion))
        respuestas.append(valor_final)
    
    try:
        for pregunta_id, valor in zip(preguntas_ids, respuestas):
            cursor.execute("""
                INSERT INTO alumno_pregunta (id_alumno, id_pregunta, valor, id_aplicacion)
                VALUES (%s, %s, %s, %s)
            """, (alumno['id'], pregunta_id, valor, id_aplicacion))
        
        promedio = sum(respuestas) / len(respuestas)
        emoji = {'bajo': '🟢', 'medio': '🟡', 'alto': '🔴'}
        print(f"  {emoji[nivel]} {alumno['nombre']} - Nivel {nivel.upper()} (Prom: {promedio:.2f})")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        conn.rollback()

conn.commit()

# ============================================
# 9. VERIFICACIÓN FINAL
# ============================================
print("\n" + "="*70)
print("✅ SETUP COMPLETADO")
print("="*70)

cursor.execute("SELECT COUNT(*) FROM usuario WHERE id NOT IN (SELECT id FROM admin)")
print(f"Alumnos creados: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM complemento")
print(f"Cuestionarios: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM alumno_pregunta")
print(f"Respuestas ansiedad: {cursor.fetchone()[0]}")

cursor.close()
conn.close()

print("\n🚀 LISTO! Ahora ejecuta:")
print("   python 05_analisis_dataset_corregido.py")
print("="*70)