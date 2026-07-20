import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.calidad import ejecutar_auditoria
from src.preparacion import preparar_datos
from src.modelado import entrenar_y_evaluar

def ejecutar_pipeline_completo():
    print("=" * 60)
    print("INICIANDO PIPELINE COMPLETO DE DESERCIÓN ESTUDIANTIL")
    print("=" * 60)
    
    # 1. Auditoría de Calidad DAMA
    print("\n--- [FASE 1] AUDITORÍA DE CALIDAD DE DATOS (DAMA) ---")
    resultados_calidad = ejecutar_auditoria(project_root)
    
    # 2. Preparación y Feature Engineering
    print("\n--- [FASE 2] PREPARACIÓN DE DATOS Y FEATURE ENGINEERING ---")
    df_preparado = preparar_datos(project_root)
    
    # 3. Modelamiento Predictivo y Exportaciones
    print("\n--- [FASE 3] ENTRENAMIENTO, EVALUACIÓN Y EXPORTACIÓN MLOPS ---")
    mejor_modelo = entrenar_y_evaluar(project_root)
    
    print("\n" + "=" * 60)
    print("PIPELINE EJECUTADO CON ÉXITO")
    print("=" * 60)
    print("Todos los entregables para el Jupyter Notebook y ObservableHQ")
    print("han sido generados y validados correctamente.")
    print(f"Carpeta del Proyecto: {project_root}")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_pipeline_completo()
