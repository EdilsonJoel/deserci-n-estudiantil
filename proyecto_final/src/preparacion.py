import pandas as pd
import numpy as np
from pathlib import Path

def aplicar_feature_engineering(df):
    """
    Aplica Feature Engineering al dataset agregando variables del rendimiento
    y estabilidad estudiantil para predecir la deserción.
    """
    df_fe = df.copy()
    
    # 1. Tasa de Aprobación Anual (Créditos aprobados / matriculados)
    total_matriculados = df_fe["Curricular units 1st sem (enrolled)"] + df_fe["Curricular units 2nd sem (enrolled)"]
    total_aprobados = df_fe["Curricular units 1st sem (approved)"] + df_fe["Curricular units 2nd sem (approved)"]
    
    df_fe["tasa_aprobacion_ano"] = np.where(
        total_matriculados > 0,
        total_aprobados / total_matriculados,
        0.0
    )
    
    # 2. Diferencial de Calificaciones (Semestre 2 - Semestre 1)
    df_fe["diferencia_calificaciones"] = df_fe["Curricular units 2nd sem (grade)"] - df_fe["Curricular units 1st sem (grade)"]
    
    # 3. Índice de Estabilidad Financiera (Becado - Deuda - Tasas retrasadas)
    df_fe["estabilidad_financiera"] = df_fe["Scholarship holder"] * 2 - (df_fe["Debtor"] + (1 - df_fe["Tuition fees up to date"]))
    
    # 4. Máxima educación alcanzada por los padres
    df_fe["educacion_padres_max"] = df_fe[["Mother's qualification", "Father's qualification"]].max(axis=1)
    
    # Mapeo ordinal de tipo_codigo para compatibilidad (usamos educacion_padres_max mapeado a escala 1-3)
    df_fe["tipo_codigo"] = np.where(df_fe["educacion_padres_max"] > 12, 3, np.where(df_fe["educacion_padres_max"] > 4, 2, 1))
    
    # Mapear Target a la columna de salida Machine failure (1 si deserta, 0 si no)
    # Conservamos el nombre Machine failure para máxima compatibilidad con scripts de modelado y ObservableHQ
    df_fe["Machine failure"] = np.where(df_fe["Target"] == "Dropout", 1, 0)
    
    return df_fe

def generar_datos_eda(df, observable_dir):
    """
    Genera resúmenes estadísticos agregados para las variables estudiantiles
    y motivos de deserción que consumirá el dashboard de ObservableHQ.
    """
    # 1. Distribución de variables continuas agregada por jornada (Daytime/evening) y Deserción
    df["Type"] = np.where(df["Daytime/evening attendance"] == 1, "Diurno", "Vespertino")
    
    variables = [
        "Age at enrollment", 
        "Curricular units 1st sem (grade)", 
        "Curricular units 2nd sem (grade)",
        "tasa_aprobacion_ano", 
        "diferencia_calificaciones", 
        "estabilidad_financiera"
    ]
    
    eda_dist = df.groupby(["Type", "Machine failure"])[variables].agg(["mean", "std", "min", "max", "count"]).reset_index()
    
    # Aplanar columnas multinivel
    eda_dist.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0] 
        for col in eda_dist.columns
    ]
    eda_dist.to_csv(observable_dir / "eda_distribuciones.csv", index=False, encoding="utf-8")
    
    # 2. Distribución jerárquica de deserciones por Carrera y Motivo para Treemap/Icicle
    desertores = df[df["Machine failure"] == 1].copy()
    
    # Mapeo de Carreras
    mapeo_carreras = {
        1: "Agronomía", 2: "Diseño y Animación", 3: "Trabajo Social", 
        4: "Tecnologías de la Info", 5: "Veterinaria", 6: "Informática", 
        7: "Turismo", 8: "Enfermería", 9: "Gestión y Administración"
    }
    # Asignar nombres legibles a las carreras, por defecto 'Otras Carreras'
    desertores["carrera_nombre"] = desertores["Course"].map(mapeo_carreras).fillna("Otras Carreras")
    
    # Clasificación de Motivo de Deserción
    motivos = []
    for idx, row in desertores.iterrows():
        if row["tasa_aprobacion_ano"] < 0.50:
            motivo = "Bajo Rendimiento Académico"
        elif row["Debtor"] == 1 or row["Tuition fees up to date"] == 0:
            motivo = "Vulnerabilidad Financiera"
        else:
            motivo = "Desadaptación/Otros"
        motivos.append(motivo)
        
    desertores["tipo_falla_especifico"] = motivos
    
    # Resumen jerárquico (Type es Diurno/Vespertino)
    jerarquia = desertores.groupby(["Type", "tipo_falla_especifico"]).size().reset_index(name="cantidad")
    # Para cumplir con el dashboard web, guardamos con el nombre original esperado
    jerarquia.to_csv(observable_dir / "eda_jerarquia_fallas.csv", index=False, encoding="utf-8")
    
    # 3. Resumen de balance de clases (Machine failure -> Deserción)
    balance = df["Machine failure"].value_counts().reset_index(name="registros")
    balance.columns = ["Machine_failure", "registros"]
    balance["porcentaje"] = (balance["registros"] / len(df) * 100).round(2)
    balance.to_csv(observable_dir / "eda_balance_clases.csv", index=False, encoding="utf-8")

def preparar_datos(project_root):
    root = Path(project_root)
    clean_path = root / "data" / "processed" / "datos_limpios.csv"
    prepared_path = root / "data" / "processed" / "datos_preparados.csv"
    observable_dir = root / "data" / "observable"
    
    print(f"Leyendo dataset limpio para Feature Engineering: {clean_path}")
    df = pd.read_csv(clean_path)
    
    # Aplicar transformaciones y nuevas variables
    df_preparado = aplicar_feature_engineering(df)
    
    # Generar archivos de soporte para EDA en Observable
    generar_datos_eda(df_preparado, observable_dir)
    
    # Guardar dataset preparado para modelamiento
    df_preparado.to_csv(prepared_path, index=False, encoding="utf-8")
    
    print("Preparación de datos y Feature Engineering de deserción completados con éxito.")
    print(f"Variables añadidas: tasa_aprobacion_ano, diferencia_calificaciones, estabilidad_financiera, educacion_padres_max, tipo_codigo")
    return df_preparado

if __name__ == "__main__":
    project_path = Path(__file__).resolve().parents[1]
    preparar_datos(project_path)
