# ¿Podemos detectar a tiempo qué estudiante va a abandonar la universidad?

**Predicción de deserción estudiantil** — Proyecto final de Big Data y Ciencia de Datos  
Universidad Nacional Arturo Jauretche (UNAJ)

---

## Descripción

Este proyecto explora si los datos del primer año —notas, situación financiera y edad— permiten identificar estudiantes en riesgo de abandono antes de que sea tarde. Se construyó un pipeline completo de Machine Learning (desde la ingesta hasta la exportación a **ONNX**) y un **dashboard interactivo en ObservableHQ** que incluye un simulador en tiempo real.

**Target:** `0` = Estable, `1` = Deserción

---

## Estructura del repositorio

```
proyecto_final/
├── data/
│   ├── raw/                  # Datos originales (UCI Student Dropout)
│   ├── processed/            # Datos limpios y preparados
│   └── observable/           # Archivos para el dashboard ObservableHQ
│       ├── *.csv             # Datos de EDA, calidad, métricas
│       ├── modelo_final.onnx # Modelo exportado a ONNX
│       └── guia_observablehq.md  # Paso a paso para construir el notebook
├── models/                   # Modelos entrenados (.joblib, .onnx)
├── notebooks/
│   └── 01_desercion_estudiantil.ipynb  # Notebook principal
├── src/
│   ├── pipeline.py           # Orquestación del pipeline
│   ├── calidad.py            # Validación de calidad DAMA
│   ├── preparacion.py        # Limpieza e ingeniería de features
│   └── modelado.py           # Entrenamiento y exportación ONNX
├── reports/figures/          # Gráficos generados
└── requirements.txt          # Dependencias Python
```

---

## Configuración del entorno

### Requisitos

- Python 3.10 – 3.13
- Git

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/EdilsonJoel/deserci-n-estudiantil.git
cd deserci-n-estudiantil

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
# source venv/bin/activate

# 3. Instalar dependencias
pip install -r proyecto_final/requirements.txt
```

### Ejecutar el notebook

```bash
jupyter notebook proyecto_final/notebooks/01_desercion_estudiantil.ipynb
```

El notebook está listo para ejecutarse de principio a fin: carga los datos, aplica calidad DAMA, entrena los modelos y exporta a ONNX.

---

## Dashboard ObservableHQ

El notebook de Observable se construye siguiendo la guía en:

```
proyecto_final/data/observable/guia_observablehq.md
```

Incluye:

- **Portada profesional** con cards de problema/datos/objetivo
- **Calidad de datos** (DAMA 100%)
- **Análisis exploratorio** con selector interactivo de variables
- **Treemap interactivo** de causas de deserción
- **Matriz de confusión** del modelo
- **Simulador en tiempo real** con ONNX (seleccionar estudiante + ajustar sliders → predicción instantánea)

### Archivos a subir

Los 13 archivos necesarios están en `proyecto_final/data/observable/`. Se suben como File Attachments en ObservableHQ.

---

## Resultados principales

| Métrica | Valor |
|---------|-------|
| Variables analizadas | 36 |
| Tasa de deserción real | 32,4 % |
| Calidad de datos (DAMA) | 100 % |
| Deserciones por bajo rendimiento | 90,6 % |

El modelo **Random Forest** alcanzó un **Recall de 95,8 %** (detecta 275 de 287 deserciones reales). La **tasa de aprobación del primer año** es la variable más influyente (54,8 % de importancia).

---

## Autor

**Edilson Joel** — UNAJ, Ingeniería  
Proyecto final de la materia Big Data y Ciencia de Datos — 2026

---

> *Demo académica con datos del dataset UCI Student Dropout. No reemplaza el criterio institucional.*
