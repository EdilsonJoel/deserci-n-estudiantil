# ¿Podemos detectar a tiempo qué estudiante va a abandonar la universidad?

**Predicción de deserción estudiantil** — Proyecto final de Big Data — UNAJ 2026

---

## ⚡ Cómo ejecutar todo (3 pasos)

### 1. Clonar y entrar

```bash
git clone https://github.com/EdilsonJoel/deserci-n-estudiantil.git
cd deserci-n-estudiantil
```

### 2. Crear entorno virtual e instalar

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
# python3 -m venv venv
# source venv/bin/activate

pip install -r proyecto_final/requirements.txt
```

### 3. Registrar el kernel (solo para Notebook)

```bash
python -m ipykernel install --user --name edaunaj2026i --display-name "Python (EdaUnaj2026I)"
```

> Si no hacés este paso, al abrir el notebook te va a aparecer "Kernel not found". Es obligatorio.

### 4. Ejecutar

```bash
# Opción A — Notebook (recomendado)
jupyter notebook proyecto_final/notebooks/01_desercion_estudiantil.ipynb

# Opción B — Script directo
python proyecto_final/src/pipeline.py
```

El pipeline completo: carga datos → valida calidad DAMA → crea features → entrena 4 modelos → exporta ONNX → genera todos los CSVs para el dashboard.

---

## 📁 Estructura del proyecto

```
deserci-n-estudiantil/
├── README.md
├── .gitignore
└── proyecto_final/
    ├── data/
    │   ├── raw/             # student_dropout.csv (UCI)
    │   ├── processed/       # datos_limpios.csv, datos_preparados.csv
    │   └── observable/      # 14 archivos para el dashboard ObservableHQ
    ├── models/              # modelo_final.joblib, .onnx, metadata.json
    ├── notebooks/
    │   └── 01_desercion_estudiantil.ipynb   ← Notebook principal
    ├── src/
    │   ├── pipeline.py      # Orquestación completa
    │   ├── calidad.py       # Validación DAMA (45 reglas)
    │   ├── preparacion.py   # Feature engineering
    │   └── modelado.py      # Entrenamiento + ONNX
    └── requirements.txt
```

---

## 📊 Dashboard en ObservableHQ

El notebook de Observable se construye manualmente siguiendo:

```
proyecto_final/data/observable/guia_observablehq.md
```

Incluye: portada profesional, calidad DAMA 100%, selector interactivo, scatterplot, treemap, matriz de confusión, simulador ONNX en tiempo real.

Los 14 archivos de `data/observable/` se suben como **File Attachments** en ObservableHQ.

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Total estudiantes | 4.424 |
| Variables analizadas | 36 |
| Tasa de deserción real | 32,4 % |
| Calidad DAMA | 100 % (45 reglas) |
| Deserciones por bajo rendimiento | 90,6 % |
| **Random Forest — Recall** | **95,8 %** |
| Random Forest — Accuracy | 94,1 % |
| Random Forest — ROC-AUC | 0,9886 |

La **tasa de aprobación del primer año** es la variable más influyente (54,8 % de importancia).

---

## 🧠 Modelos entrenados

| Modelo | Recall |
|--------|--------|
| Baseline (siempre Estable) | 0 % |
| Regresión Logística | 89,2 % |
| Naive Bayes | 88,5 % |
| **Random Forest (300 árboles)** | **95,8 %** ✅ |

---

## 📦 Requisitos

- Python 3.10 – 3.13
- pip
- Git

---

## 👤 Autor

**Edilson Joel** — UNAJ, Ingeniería — Big Data y Ciencia de Datos — 2026

---

> Demo académica con datos del dataset UCI Student Dropout. No reemplaza el criterio institucional.
