import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="anxitech"
)
cursor = conn.cursor()

print("="*70)
print("VERIFICACIÓN DE DATOS POR ALUMNO")
print("="*70)

# Query mejorada
query = """
SELECT 
    c.id_alumno,
    u.nombre,
    u.apellido,
    c.id_aplicacion,
    COUNT(DISTINCT ap.id) as respuestas_ansiedad,
    CASE 
        WHEN COUNT(DISTINCT ap.id) >= 15 THEN '✅'
        ELSE '❌'
    END as test_completo,
    '✅' as complemento_completo
FROM complemento c
INNER JOIN usuario u ON c.id_alumno = u.id
LEFT JOIN alumno_pregunta ap ON c.id_alumno = ap.id_alumno 
    AND c.id_aplicacion = ap.id_aplicacion
GROUP BY c.id_alumno, c.id_aplicacion, u.nombre, u.apellido
ORDER BY c.id_alumno
"""

cursor.execute(query)
resultados = cursor.fetchall()

print(f"\n{'ID':<6} {'Nombre':<15} {'Aplicación':<12} {'Respuestas':<12} {'Test':<8} {'Compl.':<8}")
print("-"*70)

alumnos_completos = 0
alumnos_incompletos = 0

for row in resultados:
    id_alumno, nombre, apellido, id_aplic, respuestas, test, compl = row
    print(f"{id_alumno:<6} {nombre[:15]:<15} {id_aplic:<12} {respuestas:<12} {test:<8} {compl:<8}")
    
    if respuestas >= 15:
        alumnos_completos += 1
    else:
        alumnos_incompletos += 1

print("="*70)
print(f"\n✅ Alumnos con datos COMPLETOS: {alumnos_completos}")
print(f"❌ Alumnos con datos INCOMPLETOS: {alumnos_incompletos}")

if alumnos_incompletos > 0:
    print(f"\n⚠️  Hay {alumnos_incompletos} alumnos que necesitan respuestas de ansiedad")
    print("   Ejecuta el script corregido que te daré a continuación.")
else:
    print("\n✅ Todos los alumnos tienen datos completos")
    print("   El problema está en el script de análisis.")

cursor.close()
conn.close()