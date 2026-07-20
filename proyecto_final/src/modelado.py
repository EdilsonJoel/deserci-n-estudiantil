import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)

# Serializers
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

def entrenar_y_evaluar(project_root):
    root = Path(project_root)
    prepared_path = root / "data" / "processed" / "datos_preparados.csv"
    models_dir = root / "models"
    observable_dir = root / "data" / "observable"
    figures_dir = root / "reports" / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    observable_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando datos preparados para modelado: {prepared_path}")
    df = pd.read_csv(prepared_path)
    
    # ----------------------------------------------------
    # Definición de variables X e y (Control de Data Leakage)
    # ----------------------------------------------------
    # Predictores (fáciles de medir u observar en primer año escolar)
    features = [
        "tipo_codigo",              # Nivel educativo de los padres mapeado
        "Age at enrollment",
        "Gender",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "tasa_aprobacion_ano",       # Feature Engineering
        "diferencia_calificaciones", # Feature Engineering
        "estabilidad_financiera"     # Feature Engineering
    ]
    
    # Target (representa deserción: 1 si es Dropout, 0 si no)
    target = "Machine failure"
    
    X = df[features].copy()
    y = df[target].copy()
    
    # Convertimos todo X a float32 para simplificar la exportación a ONNX
    X = X.astype(np.float32)
    
    # Split Train/Test estratificado (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    print(f"Registros en Train: {X_train.shape[0]} (Deserciones: {y_train.sum()} - {y_train.mean()*100:.2f}%)")
    print(f"Registros en Test: {X_test.shape[0]} (Deserciones: {y_test.sum()} - {y_test.mean()*100:.2f}%)")
    
    # ----------------------------------------------------
    # Preprocesamiento reproducible
    # ----------------------------------------------------
    preprocessing_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # ----------------------------------------------------
    # Configuración de Modelos
    # ----------------------------------------------------
    # 1. Baseline: Dummy Classifier que predice la clase mayoritaria (no desertores)
    baseline = DummyClassifier(strategy="most_frequent")
    
    # 2. Regresión Logística con balance de pesos de clases
    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=42
    )
    
    # 3. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Naive Bayes (GaussianNB)
    nb = GaussianNB()
    
    modelos = {
        "Baseline (Mayoritaria)": baseline,
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Naive Bayes": nb
    }
    
    # Entrenamiento y Evaluación
    metricas = []
    fitted_pipelines = {}
    
    for nombre, model in modelos.items():
        # Crear pipeline completo
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessing_pipeline),
            ("classifier", model)
        ])
        
        # Entrenar pipeline
        pipe.fit(X_train.to_numpy(), y_train)
        fitted_pipelines[nombre] = pipe
        
        # Predecir y calcular probabilidades
        preds = pipe.predict(X_test.to_numpy())
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test.to_numpy())[:, 1]
        else:
            probs = preds
            
        # Calcular métricas
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        bal_acc = balanced_accuracy_score(y_test, preds)
        
        metricas.append({
            "modelo": nombre,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "balanced_accuracy": round(bal_acc, 4)
        })
        
    metricas_df = pd.DataFrame(metricas).sort_values("f1", ascending=False)
    metricas_df.to_csv(observable_dir / "metricas_modelos.csv", index=False, encoding="utf-8")
    
    print("\nResultados de evaluación en Test (Ordenados por F1-Score):")
    print(metricas_df.to_string(index=False))
    
    # ----------------------------------------------------
    # Exportación del mejor modelo
    # ----------------------------------------------------
    mejor_modelo_nombre = metricas_df.iloc[0]["modelo"]
    mejor_f1 = metricas_df.iloc[0]["f1"]
    print(f"\nEl mejor modelo es '{mejor_modelo_nombre}' con un F1-Score de {mejor_f1:.4f}")
    
    mejor_pipeline = fitted_pipelines[mejor_modelo_nombre]
    
    # 1. Guardar Pipeline final en Joblib (.joblib)
    joblib_path = models_dir / "modelo_final.joblib"
    joblib.dump(mejor_pipeline, joblib_path)
    print(f"Modelo final en Joblib guardado en: {joblib_path}")
    
    # 2. Generar Matriz de Confusión para el mejor modelo
    mejor_preds = mejor_pipeline.predict(X_test.to_numpy())
    cm = confusion_matrix(y_test, mejor_preds)
    cm_df = pd.DataFrame({
        "prediccion": ["0 (Estable)", "0 (Estable)", "1 (Deserción)", "1 (Deserción)"],
        "real": ["0 (Estable)", "1 (Deserción)", "0 (Estable)", "1 (Deserción)"],
        "cantidad": [cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]]
    })
    cm_df.to_csv(observable_dir / "matriz_confusion.csv", index=False, encoding="utf-8")
    
    # 3. Importancia de Variables (si el mejor modelo es Random Forest)
    if "Random Forest" in mejor_modelo_nombre:
        importancias = mejor_pipeline.named_steps["classifier"].feature_importances_
        imp_df = pd.DataFrame({
            "variable": features,
            "importancia": importancias
        }).sort_values("importancia", ascending=False)
        imp_df.to_csv(observable_dir / "importancia_variables.csv", index=False, encoding="utf-8")
        print("\nImportancia de Variables (Random Forest):")
        print(imp_df.to_string(index=False))
        
    # ----------------------------------------------------
    # MLOps: Exportación a ONNX
    # ----------------------------------------------------
    # Exportamos la Regresión Logística a ONNX por estabilidad de probabilidades en el navegador
    onnx_path = models_dir / "modelo_final.onnx"
    pipeline_onnx = fitted_pipelines["Logistic Regression"]
    
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    
    try:
        options = {'classifier': {'zipmap': False}}
        onx = convert_sklearn(pipeline_onnx, initial_types=initial_type, target_opset=15, options=options)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"Modelo de Regresión Logística compatible exportado a ONNX en: {onnx_path}")
        
        # Validar consistencia del modelo ONNX contra Scikit-Learn
        sess = rt.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        
        sample_x = X_test.iloc[:5].to_numpy().astype(np.float32)
        sk_preds = pipeline_onnx.predict(sample_x)
        onnx_preds = sess.run(None, {input_name: sample_x})[0]
        
        print("\nValidación de Consistencia ONNX (Regresión Logística):")
        print(f"Muestras de entrada:\n{sample_x}")
        print(f"Predicciones Scikit-Learn (LR) : {sk_preds}")
        print(f"Predicciones ONNX              : {onnx_preds}")
        
        if np.array_equal(sk_preds, onnx_preds):
            print(">>> VALIDACIÓN EXITOSA: Las predicciones de ONNX coinciden exactamente con Scikit-Learn.")
        else:
            print(">>> ADVERTENCIA: Hay discrepancias numéricas menores entre ONNX y Scikit-Learn.")
            
    except Exception as e:
        print(f"Error al exportar o validar en formato ONNX: {e}")
        
    # ----------------------------------------------------
    # Generar muestras de test para simulación en la web
    # ----------------------------------------------------
    muestra_test = X_test.copy()
    muestra_test["Machine_failure_real"] = y_test
    muestra_test["Machine_failure_pred"] = pipeline_onnx.predict(X_test.to_numpy())
    
    mapeo_inverso = {1: "Básica", 2: "Secundaria", 3: "Superior"}
    muestra_test["Type"] = muestra_test["tipo_codigo"].map(mapeo_inverso)
    
    # Exportar las primeras 100 muestras
    muestra_test = pd.concat([
        muestra_test[muestra_test["Machine_failure_real"] == 1].head(30),
        muestra_test[muestra_test["Machine_failure_real"] == 0].head(70)
    ]).sample(frac=1.0, random_state=42)
    
    muestra_test = muestra_test.reset_index().rename(columns={"index": "id_maquina"})
    muestra_test.to_csv(observable_dir / "predicciones_modelo.csv", index=False, encoding="utf-8")
    
    # 4. Exportar límites del simulador
    limites = {}
    for col in features:
        limites[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median())
        }
    with open(observable_dir / "modelo_web_config.json", "w", encoding="utf-8") as f:
        json.dump(limites, f, indent=4)
        
    # Exportar metadatos generales
    metadatos = {
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mejor_modelo": mejor_modelo_nombre,
        "f1_score": float(mejor_f1),
        "total_filas_dataset": int(len(df)),
        "variables_entrada": features
    }
    with open(models_dir / "modelo_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4)
        
    pd.DataFrame([metadatos]).to_csv(observable_dir / "metadata_proyecto.csv", index=False, encoding="utf-8")
    
    print("\nModelado predictivo de deserción e inferencias de simulación completadas con éxito.")
    return mejor_pipeline

if __name__ == "__main__":
    project_path = Path(__file__).resolve().parents[1]
    entrenar_y_evaluar(project_path)
