        # 02_entrenar_modelo.py - Entrenar Random Forest CON OPCIONES REALES
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("=== ENTRENANDO MODELO RANDOM FOREST ===")

        # 1. CARGAR DATOS
        print("Cargando datos...")
        dataset = pd.read_csv('datos/dataset_completo.csv')
        print(f"Dataset cargado: {len(dataset)} registros")
        print(f"Columnas disponibles: {list(dataset.columns)}")

        # 2. PREPARAR VARIABLES
        print("Preparando variables...")

        # Variables categóricas a números
        le_carrera = LabelEncoder()
        le_transporte = LabelEncoder()
        le_hogar = LabelEncoder()  # Cambiado de "convivencia" a "hogar"

        dataset['carrera_num'] = le_carrera.fit_transform(dataset['carrera'])
        dataset['transporte_num'] = le_transporte.fit_transform(dataset['transporte'])
        dataset['hogar_num'] = le_hogar.fit_transform(dataset['hogar'])  # Actualizado

        # Mostrar mapeos para referencia
        print("\n Mapeos de variables categóricas:")
        print("Carreras:")
        for i, carrera in enumerate(le_carrera.classes_):
            print(f"  {i}: {carrera}")

        print("\nTransportes:")
        for i, transporte in enumerate(le_transporte.classes_):
            print(f"  {i}: {transporte}")

        print("\nHogar:")
        for i, hogar in enumerate(le_hogar.classes_):
            print(f"  {i}: {hogar}")

        # Variables predictoras 
        features = [
            'promedio',
            'semestre', 
            'num_materias',
            'carrera_num',
            'transporte_num', 
            'hogar_num',    
            'trabaja',
            'beca'
        ]

        X = dataset[features]
        y = dataset['nivel_ansiedad']

        print(f"\nVariables predictoras: {features}")
        print(f"Variable objetivo: nivel_ansiedad")
        print(f"Forma de X: {X.shape}")
        print(f"Distribución de y: {y.value_counts()}")

        # 3. DIVIDIR DATOS
        print("\n Dividiendo datos...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Entrenamiento: {len(X_train)} registros")
        print(f"Prueba: {len(X_test)} registros")

        # 4. ENTRENAR RANDOM FOREST
        print("\n Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,      # 100 árboles
            max_depth=10,          # Profundidad máxima
            min_samples_split=5,   # Mínimo para dividir
            min_samples_leaf=2,    # Mínimo en hojas
            random_state=42
        )

        rf_model.fit(X_train, y_train)
        print(" Modelo entrenado")

        # 5. HACER PREDICCIONES
        print(" Haciendo predicciones...")
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)

        # 6. EVALUAR MODELO
        print("\n Evaluando modelo...")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión general: {accuracy:.3f} ({accuracy*100:.1f}%)")

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # 7. IMPORTANCIA DE VARIABLES
        print("\n Importancia de variables:")
        importancia = pd.DataFrame({
            'Variable': features,
            'Importancia': rf_model.feature_importances_
        }).sort_values('Importancia', ascending=False)

        print(importancia.round(4))

        # 8. CREAR VISUALIZACIONES
        print("\n Creando visualizaciones...")

        # Configurar estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], 
                    xticklabels=['Alto', 'Bajo', 'Medio'],
                    yticklabels=['Alto', 'Bajo', 'Medio'],
                    cmap='Blues')
        axes[0,0].set_title('Matriz de Confusión')
        axes[0,0].set_xlabel('Predicción')
        axes[0,0].set_ylabel('Realidad')

        # Importancia de variables
        axes[0,1].barh(importancia['Variable'], importancia['Importancia'], color='skyblue')
        axes[0,1].set_title('Importancia de Variables')
        axes[0,1].set_xlabel('Importancia')

        # Distribución real vs predicha
        nivel_counts = dataset['nivel_ansiedad'].value_counts()
        axes[1,0].bar(nivel_counts.index, nivel_counts.values, alpha=0.7, color='lightcoral')
        axes[1,0].set_title('Distribución Real de Ansiedad')
        axes[1,0].set_ylabel('Número de Estudiantes')

        # Predicciones por clase
        pred_df = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})
        pred_counts = pred_df['Predicho'].value_counts()
        axes[1,1].bar(pred_counts.index, pred_counts.values, alpha=0.7, color='orange')
        axes[1,1].set_title('Distribución Predicha')
        axes[1,1].set_ylabel('Número de Estudiantes')

        plt.tight_layout()
        plt.savefig('resultados/evaluacion_modelo_real.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 9. ANÁLISIS POR CARRERA
        print("\n📊 Análisis por carrera:")
        carrera_ansiedad = dataset.groupby('carrera')['dass21_score'].agg(['mean', 'std', 'count']).round(2)
        carrera_ansiedad = carrera_ansiedad.sort_values('mean', ascending=False)
        print(carrera_ansiedad)

        # 10. GUARDAR MODELO Y ENCODERS
        print("\n Guardando modelo y encoders...")
        import joblib
        os.makedirs('modelos', exist_ok=True)

        joblib.dump(rf_model, 'modelos/random_forest_ansiedad.pkl')
        joblib.dump(le_carrera, 'modelos/encoder_carrera.pkl')
        joblib.dump(le_transporte, 'modelos/encoder_transporte.pkl')
        joblib.dump(le_hogar, 'modelos/encoder_hogar.pkl')

        # Guardar también los mapeos para referencia
        mapeos = {
            'carreras': {carrera: i for i, carrera in enumerate(le_carrera.classes_)},
            'transportes': {transporte: i for i, transporte in enumerate(le_transporte.classes_)},
            'hogares': {hogar: i for i, hogar in enumerate(le_hogar.classes_)}
        }

        joblib.dump(mapeos, 'modelos/mapeos_variables.pkl')

        print(" Modelo y encoders guardados en 'modelos/'")

        # 11. EJEMPLO DE PREDICCIÓN CON OPCIONES REALES
        print("\n EJEMPLO DE PREDICCIÓN CON OPCIONES REALES:")
        print("Nuevo estudiante del ITO:")
        nuevo_estudiante = {
            'promedio': 7.5,
            'semestre': 5,
            'num_materias': 6,
            'carrera': 'En sistemas computacionales',
            'transporte': 'Transporte publico',
            'hogar': 'Con familiares', 
            'trabaja': 1,  # Sí trabaja
            'beca': 0      # No tiene beca
        }

        # Transformar a formato del modelo
        nuevo_X = np.array([[
            nuevo_estudiante['promedio'],
            nuevo_estudiante['semestre'],
            nuevo_estudiante['num_materias'],
            le_carrera.transform([nuevo_estudiante['carrera']])[0],
            le_transporte.transform([nuevo_estudiante['transporte']])[0],
            le_hogar.transform([nuevo_estudiante['hogar']])[0],
            nuevo_estudiante['trabaja'],
            nuevo_estudiante['beca']
        ]])

        prediccion = rf_model.predict(nuevo_X)[0]
        probabilidades = rf_model.predict_proba(nuevo_X)[0]

        print(f"\nCaracterísticas del estudiante:")
        for key, value in nuevo_estudiante.items():
            print(f"  {key}: {value}")

        print(f"\nPREDICCIÓN: {prediccion}")
        print(f"Probabilidades:")
        clases = rf_model.classes_
        for i, clase in enumerate(clases):
            print(f"  {clase}: {probabilidades[i]:.3f} ({probabilidades[i]*100:.1f}%)")

        print(f"\nConfianza de la predicción: {max(probabilidades):.3f} ({max(probabilidades)*100:.1f}%)")

        print("\n ¡MODELO COMPLETADO!")
