import pandas as pd
import numpy as np
from pathlib import Path

def evaluar_calidad(df):
    """
    Evalúa las dimensiones de calidad de datos bajo el marco DAMA
    para el dataset UCI Student Dropout.
    """
    total_registros = len(df)
    
    # ----------------------------------------------------
    # 1. Completitud: Porcentaje de registros no nulos
    # ----------------------------------------------------
    nulos = df.isnull().sum()
    calidad_variables = []
    
    for col in df.columns:
        nulos_col = int(nulos[col])
        completitud_pct = (1 - (nulos_col / total_registros)) * 100
        calidad_variables.append({
            "columna": col,
            "dimension": "Completitud",
            "metrica": "Porcentaje de completitud",
            "valor": round(completitud_pct, 4),
            "estado": "OK" if completitud_pct == 100.0 else "OBSERVAR"
        })

    # ----------------------------------------------------
    # 2. Unicidad: Duplicados en claves primarias
    # ----------------------------------------------------
    duplicados_student_id = int(df["Student_ID"].duplicated().sum())
    
    reglas_detalle = [
        {
            "dimension": "Unicidad",
            "regla": "Clave Student_ID única",
            "registros_evaluados": total_registros,
            "registros_incumplen": duplicados_student_id,
            "estado": "OK" if duplicados_student_id == 0 else "ERROR"
        }
    ]

    # ----------------------------------------------------
    # 3. Consistencia: Coherencia lógica y física
    # ----------------------------------------------------
    # Regla 3.1: Créditos aprobados <= Créditos matriculados
    inconsistencia_sem1 = int((df["Curricular units 1st sem (approved)"] > df["Curricular units 1st sem (enrolled)"]).sum())
    inconsistencia_sem2 = int((df["Curricular units 2nd sem (approved)"] > df["Curricular units 2nd sem (enrolled)"]).sum())
    
    # Regla 3.2: Notas en rango de 0 a 20
    notas_invalidas_sem1 = int(((df["Curricular units 1st sem (grade)"] < 0) | (df["Curricular units 1st sem (grade)"] > 20)).sum())
    notas_invalidas_sem2 = int(((df["Curricular units 2nd sem (grade)"] < 0) | (df["Curricular units 2nd sem (grade)"] > 20)).sum())
    
    # Regla 3.3: Edad coherente (> 16 años)
    edad_incoherente = int((df["Age at enrollment"] < 16).sum())
    
    reglas_detalle.extend([
        {
            "dimension": "Consistencia",
            "regla": "Créditos aprobados <= matriculados Sem1",
            "registros_evaluados": total_registros,
            "registros_incumplen": inconsistencia_sem1,
            "estado": "OK" if inconsistencia_sem1 == 0 else "ERROR"
        },
        {
            "dimension": "Consistencia",
            "regla": "Créditos aprobados <= matriculados Sem2",
            "registros_evaluados": total_registros,
            "registros_incumplen": inconsistencia_sem2,
            "estado": "OK" if inconsistencia_sem2 == 0 else "ERROR"
        },
        {
            "dimension": "Consistencia",
            "regla": "Notas en rango [0, 20]",
            "registros_evaluados": total_registros,
            "registros_incumplen": notas_invalidas_sem1 + notas_invalidas_sem2,
            "estado": "OK" if (notas_invalidas_sem1 + notas_invalidas_sem2) == 0 else "ERROR"
        },
        {
            "dimension": "Consistencia",
            "regla": "Edad de matrícula >= 16",
            "registros_evaluados": total_registros,
            "registros_incumplen": edad_incoherente,
            "estado": "OK" if edad_incoherente == 0 else "ERROR"
        }
    ])

    # ----------------------------------------------------
    # 4. Validez: Rangos de datos y dominios válidos
    # ----------------------------------------------------
    # Regla 4.1: Banderas binarias válidas (0 o 1)
    binarios_invalidos = 0
    columnas_binarias = ["Gender", "Scholarship holder", "Debtor", "Tuition fees up to date", "Daytime/evening attendance", "International"]
    for col in columnas_binarias:
        binarios_invalidos += int((~df[col].isin([0, 1])).sum())
        
    # Regla 4.2: Targets válidos ('Dropout', 'Graduate', 'Enrolled')
    targets_invalidos = int((~df["Target"].isin(["Dropout", "Graduate", "Enrolled"])).sum())

    reglas_detalle.extend([
        {
            "dimension": "Validez",
            "regla": "Banderas binarias en [0, 1]",
            "registros_evaluados": total_registros,
            "registros_incumplen": binarios_invalidos,
            "estado": "OK" if binarios_invalidos == 0 else "ERROR"
        },
        {
            "dimension": "Validez",
            "regla": "Target en [Dropout, Graduate, Enrolled]",
            "registros_evaluados": total_registros,
            "registros_incumplen": targets_invalidos,
            "estado": "OK" if targets_invalidos == 0 else "ERROR"
        }
    ])

    # ----------------------------------------------------
    # Agrupación por dimensiones
    # ----------------------------------------------------
    reglas_df = pd.DataFrame(reglas_detalle)
    
    resumen_dimensiones = reglas_df.groupby("dimension").agg(
        reglas_totales=("regla", "count"),
        reglas_aprobadas=("estado", lambda s: (s == "OK").sum()),
        registros_evaluados=("registros_evaluados", "first"),
        total_incumplimientos=("registros_incumplen", "sum")
    ).reset_index()
    
    # Agregar completitud al resumen
    completitud_promedio = pd.DataFrame(calidad_variables)["valor"].mean()
    resumen_dimensiones = pd.concat([
        resumen_dimensiones,
        pd.DataFrame([{
            "dimension": "Completitud",
            "reglas_totales": len(df.columns),
            "reglas_aprobadas": (pd.DataFrame(calidad_variables)["estado"] == "OK").sum(),
            "registros_evaluados": total_registros,
            "total_incumplimientos": int(nulos.sum())
        }])
    ], ignore_index=True)
    
    # Calcular tasa de cumplimiento por dimensión
    resumen_dimensiones["cumplimiento_pct"] = (resumen_dimensiones["reglas_aprobadas"] / resumen_dimensiones["reglas_totales"]) * 100
    resumen_dimensiones["cumplimiento_pct"] = resumen_dimensiones["cumplimiento_pct"].round(2)
    
    return {
        "calidad_variables": pd.DataFrame(calidad_variables),
        "reglas_calidad_detalle": reglas_df,
        "calidad_dimensiones": resumen_dimensiones
    }

def ejecutar_auditoria(project_root):
    root = Path(project_root)
    raw_path = root / "data" / "raw" / "student_dropout.csv"
    processed_dir = root / "data" / "processed"
    observable_dir = root / "data" / "observable"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    observable_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Leyendo dataset crudo para auditoría DAMA: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Agregar clave primaria secuencial Student_ID si no existe
    if "Student_ID" not in df.columns:
        df["Student_ID"] = range(1001, 1001 + len(df))
        
    # Reordenar para poner Student_ID al inicio
    cols = ["Student_ID"] + [col for col in df.columns if col != "Student_ID"]
    df = df[cols]
    
    # Ejecutar evaluación
    resultados = evaluar_calidad(df)
    
    # Guardar reportes en observable
    resultados["calidad_variables"].to_csv(observable_dir / "calidad_variables.csv", index=False, encoding="utf-8")
    resultados["reglas_calidad_detalle"].to_csv(observable_dir / "reglas_calidad_detalle.csv", index=False, encoding="utf-8")
    resultados["calidad_dimensiones"].to_csv(observable_dir / "calidad_dimensiones.csv", index=False, encoding="utf-8")
    
    # Guardar datos limpios procesados
    df.to_csv(processed_dir / "datos_limpios.csv", index=False, encoding="utf-8")
    
    print("Auditoría de calidad DAMA finalizada con éxito.")
    print("Reportes guardados en la carpeta data/observable/.")
    print(resultados["calidad_dimensiones"])
    return resultados

if __name__ == "__main__":
    project_path = Path(__file__).resolve().parents[1]
    ejecutar_auditoria(project_path)
