# Guía de presentación oral — 12 minutos

**Proyecto:** ¿Podemos detectar a tiempo qué estudiante va a abandonar la universidad?
**Autor:** Edilson Joel — UNAJ, Big Data y Ciencia de Datos — 2026
**Dataset:** UCI Student Dropout (Portugal, 4,424 estudiantes, 36 variables)
**Dashboard:** ObservableHQ | **Pipeline:** Python + ONNX

---

## [0:00—0:35] HOOK + APERTURA (Contexto del negocio)

*Pararse frente al dashboard. Mostrar la portada con el título grande y las 3 cards.*

**Habla:** «Buenos días a todos. Les voy a hacer una pregunta: ¿cuántos de ustedes conocen a alguien que empezó la universidad y dejó? Seguro más de uno. En la facu vemos compañeros que desaparecen después del primer parcial, y la pregunta de mi proyecto es justamente esa: **¿podemos detectar a tiempo qué estudiante va a abandonar?**».

*Señalar con el mouse las 3 cards: Problema, Datos, Objetivo.*

**Habla:** «Este proyecto trabajó con datos reales de 4.424 estudiantes de una universidad de Portugal —el dataset UCI Student Dropout— con 36 variables: notas, situación financiera, edad, si tiene beca, si es deudor, etc. El objetivo fue construir un modelo que clasifique a cada estudiante en **Estable (0)** o **Deserción (1)** , y meterlo en un simulador interactivo».

*Señalar la blockquote azul.*

**Habla:** «La pregunta rectora fue: ¿en qué medida las variables socioeconómicas y el rendimiento del primer año permiten predecir la deserción?».

---

## [0:35—1:15] COMPRENSIÓN DE LOS DATOS (Stats cards + Calidad DAMA)

*Bajar a las 4 stats cards.*

**Habla:** «Primero, ¿con qué datos contamos? Acá un resumen rápido».

*Mouse sobre cada card:*

1. **«36 Variables»** — «Originales más las que creamos por ingeniería de atributos, como tasa de aprobación anual o estabilidad financiera».
2. **«32.4%»** — «Ese es el desbalance: de los 4.424 estudiantes, el 32.4% desertó. No es un desbalance extremo, pero hay que tenerlo en cuenta al entrenar».
3. **«100% DAMA»** — «Aplicamos el marco de calidad DAMA con 45 reglas sobre completitud, consistencia, validez y unicidad. Pasamos todas».
4. **«ONNX»** — «Exportamos el modelo a formato ONNX para que corra directamente en el navegador, sin servidores».

*Bajar al gráfico DAMA.*

**Habla:** «Y acá está el detalle de calidad: cuatro dimensiones —Completitud, Consistencia, Validez, Unicidad— todas al 100%. Esto nos dio confianza de que los datos estaban listos para modelar».

*Pausa breve. Señalar la tabla de reglas.*

**Habla:** «Abajo, la tabla muestra cada regla validada. No encontramos ningún registro fuera de especificación».

---

## [1:15—2:45] PREPARACIÓN + ANÁLISIS EXPLORATORIO (CRISP-DM: Preparación)

### Balance de clases [1:15—1:40]

*Señalar el gráfico de barras.*

**Habla:** «Miramos la variable objetivo: 67.5% estables frente a 32.5% de deserción. Ya ven que no hay empate técnico, pero el 32% es lo suficientemente alto como para que valga la pena modelarlo».

### Selector interactivo + barras [1:40—2:15]

*Cambiar el selector a "Tasa de aprobación anual".*

**Habla:** «Acá está lo interesante. Este selector permite comparar cualquier variable entre los dos grupos».

*Mouse sobre las barras.*

**Habla:** «Miren la tasa de aprobación anual: los estables promedian alrededor del 70%, los que desertan apenas llegan al 30%. Esa brecha es enorme».

*Cambiar el selector a "Edad de matrícula".*

**Habla:** «Si cambiamos a edad, vemos que los estudiantes que desertan son ligeramente mayores al ingresar. No es tan marcado como las notas, pero aparece».

### Scatterplot [2:15—2:35]

*Señalar el scatterplot.*

**Habla:** «Este gráfico cruza edad con tasa de aprobación. Cada punto es un estudiante. Los azules son los estables, los rojos los que desertaron».

*Dibujar un círculo imaginario con el mouse sobre la zona roja.*

**Habla:** «Fíjense: los puntos rojos se concentran abajo a la derecha —baja aprobación, mayor edad— y también abajo a la izquierda —baja aprobación, cualquier edad. La tasa de aprobación es el verdadero divisor de aguas».

### Insight blockquote [2:35—2:45]

*Señalar la cita azul.*

**Habla:** «Para no dejar dudas, el insight lo dice claro: los desertores tienen menor tasa de aprobación, menor estabilidad financiera y son mayores. Esto se repite tanto en diurno como en vespertino».

---

## [2:45—3:45] TREEMAP: ¿POR QUÉ DESERTAN?

*Señalar el treemap.*

**Habla:** «Ahora: de los que se van, ¿por qué se van? Acá lo dividimos en tres grandes motivos».

*Pasar el mouse sobre cada bloque del treemap:*

1. **Rojo, gigante: «Bajo Rendimiento Académico — 90.6%»** — «Casi 9 de cada 10 deserciones se explican por bajo rendimiento. Es aplastante».
2. **Naranja, pequeño: «Vulnerabilidad Financiera — 6.5%»** — «Problemas de deuda, becas, morosidad. Existe, pero es mucho menor».
3. **Gris, mínimo: «Desadaptación/Otros — 2.9%»** — «Casos aislados de adaptación, cambio de carrera, etc.».

*Hacer zoom mental.*

**Habla:** «Esto confirma lo que veíamos en el scatter: el rendimiento académico del primer año es el factor crítico. Si un alumno arranca mal, el riesgo se dispara».

---

## [3:45—5:45] MODELADO + EVALUACIÓN (CRISP-DM: Modelado y Evaluación)

### Título + qué entrenamos [3:45—4:10]

*Señalar el título de la sección 4.*

**Habla:** «Pasamos a la pregunta clave: ¿qué tan bien detecta el riesgo nuestro modelo? Entrenamos 4 modelos: una línea de base que siempre predice la clase mayoritaria, regresión logística, Random Forest y Naive Bayes».

### Tabla de métricas [4:10—4:40]

*Señalar la tabla.*

**Habla:** «Comparamos Accuracy, Precisión, Recall, F1 y ROC-AUC. El que mejor rindió fue **Random Forest con un Recall de 95.8%** —detecta 275 de 287 deserciones reales— y un Accuracy de 94.1%».

*Señalar la fila de Baseline.*

**Habla:** «La baseline, que sería no hacer nada y decir que todos son estables, tiene 67.5% de accuracy... pero Recall 0%. No detecta una sola deserción. Eso muestra por qué el accuracy solo no sirve en problemas desbalanceados».

### Matriz de confusión [4:40—5:20]

*Señalar el heatmap.*

**Habla:** «La matriz de confusión del Random Forest nos cuenta la historia completa».

*Señalar cada celda:*

1. **«558 — Estables bien clasificados»** — «La mayoría de los estables, bien».
2. **«275 — Deserciones bien detectadas»** — «El modelo atrapa 275 de 287 deserciones reales».
3. **«12 — Falsos Negativos»** — «Estos son los que más nos duelen: 12 estudiantes que desertaron pero el modelo dijo que eran estables. Son los que se van sin que nadie los alerte».
4. **«40 — Falsos Positivos»** — «Y acá 40 estables que el modelo marcó como riesgo. Falsa alarma, pero el costo es bajo: una tutoría preventiva».

### Interpretation cards [5:20—5:45]

*Señalar las 3 cards de colores (verde, rojo, naranja).*

**Habla:** «Resumiendo los costos operativos: tenemos **833 aciertos** (verde), **12 falsos negativos** (rojo, costo altísimo porque el alumno se va sin intervención) y **40 falsos positivos** (naranja, costo bajo: una charla con tutores). Priorizamos Recall sobre Precisión justamente por esto: preferimos 40 falsas alarmas antes que perder a un solo estudiante sin haberlo intentado».

---

## [5:45—9:00] SIMULADOR ONNX: 3 DEMOS (Visualización)

*Subir el tono, es la parte más interactiva.*

**Habla:** «Y ahora la parte que más me gusta: el simulador. Corre Machine Learning en el navegador con ONNX, sin servidores, sin internet. Puedo seleccionar un estudiante de la tabla y ajustar sus datos con sliders».

---

### Demo 1: Alto riesgo [5:55—6:55]

**Habla:** «Voy a simular un estudiante con **alta probabilidad de deserción**. Elijan uno cualquiera de la tabla o directamente movemos los sliders».

1. *Mover slider Edad a 35+.* **Habla:** «Le pongo 35 años, un alumno que entra más grande».
2. *Mover slider Tasa de aprobación a 15%.* **Habla:** «Tasa de aprobación bajísima: 15% de materias aprobadas en el primer año. Esto ya es bandera roja».
3. *Mover slider Estabilidad financiera a -4.0.* **Habla:** «Y estabilidad financiera negativa: arrastra deudas, no está al día con aranceles».

*Pausa dramática. Señalar el resultado.*

**Habla:** «Miren el resultado: **⚠ Riesgo de deserción. Probabilidad: 94%, 95%, lo que dé**. El modelo lo marca claramente. Este es el perfil de alumno que necesita intervención urgente».

---

### Demo 2: Bajo riesgo [6:55—7:45]

**Habla:** «Ahora el caso opuesto: un estudiante sin riesgo».

1. *Mover slider Edad a 18.* **Habla:** «Edad típica de ingreso: 18 años».
2. *Mover slider Tasa de aprobación a 95%.* **Habla:** «Tasa de aprobación altísima: 95%, prácticamente aprueba todo».
3. *Mover slider Estabilidad financiera a 4.5.* **Habla:** «Y estabilidad financiera positiva: becado, sin deudas, al día».

*Señalar el resultado verde.*

**Habla:** «**✅ Estudiante estable. Probabilidad de deserción: menos del 5%.** Este alumno probablemente se gradúa sin problemas».

---

### Demo 3: Caso intermedio — el "normal" [7:45—8:45]

**Habla:** «Y el caso más común, el estudiante del medio, ese que tiene altibajos».

1. *Mover slider Edad a 20.* **Habla:** «20 años, edad normal».
2. *Mover slider Tasa de aprobación a 55%.* **Habla:** «Tasa de aprobación del 55%: aprueba un poco más de la mitad de las materias».
3. *Mover slider Estabilidad financiera a 0.* **Habla:** «Estabilidad financiera neutra, cero: ni bien ni mal».

*Señalar el resultado.*

**Habla:** «El resultado: **probabilidad intermedia, digamos 40-50%.** Es un caso límite. Con un trimestre malo podría caer a deserción. Acá está el valor del sistema: detectar a estos estudiantes a tiempo, cuando todavía se puede actuar».

---

### Explicación técnica breve [8:45—9:00]

**Habla:** «Técnicamente, el modelo recibe 9 variables numéricas, corre una inferencia con ONNX Runtime Web, y devuelve la clase y la probabilidad. Todo en milisegundos, directo en el navegador».

---

## [9:00—10:30] CONCLUSIONES + RECOMENDACIÓN + LIMITACIONES

### Conclusiones [9:00—9:45]

*Señalar la sección "¿Qué aprendimos?".*

**Habla:** «¿Qué nos llevamos de este proyecto? Cuatro cosas».

1. *Señalar el primer punto.* **«La tasa de aprobación del primer año es la variable más determinante: concentra el 54.8% de importancia en el modelo. Si un alumno reprueba desde el inicio, el riesgo es altísimo».**
2. *Señalar el segundo.* **«La estabilidad financiera y la edad también pesan, pero mucho menos que el rendimiento».**
3. *Señalar el tercero.* **«El treemap lo confirmó: 9 de cada 10 deserciones son por bajo rendimiento. No es principalmente un problema económico, es académico».**
4. *Señalar el cuarto.* **«Logramos que el modelo corra en el navegador con ONNX. Esto abre la puerta a herramientas accesibles sin infraestructura pesada».**

### Recomendación [9:45—10:05]

**Habla:** «Mi recomendación concreta: implementar un sistema de alertas tempranas que monitoree la tasa de aprobación del primer semestre. Si un estudiante tiene menos del 50% de materias aprobadas al cierre del primer cuatrimestre, debería recibir tutoría académica automática y una evaluación financiera antes del segundo semestre».

### Limitaciones [10:05—10:30]

**Habla:** «Pero ojo, esto tiene limitaciones importantes».

1. **«Los datos son de Portugal, no de Perú. Los patrones pueden ser distintos en nuestro contexto».**
2. **«Solo usamos datos del primer año. No considera crisis económicas, cambios de política educativa o problemas personales que surjan después».**
3. **«La variable objetivo es binaria: deserta o no deserta. No captura matices como abandono temporal, cambio de carrera, o si el estudiante vuelve después».**

---

## [10:30—12:00] CIERRE + RONDA DE PREGUNTAS

*Volver a la portada o a la blockquote final.*

**Habla:** «Para cerrar: ¿podemos detectar a tiempo qué estudiante va a abandonar la universidad? La respuesta es **sí, con buena精确itud**. Con solo mirar las notas del primer año y algunas variables financieras podemos identificar al 95% de los estudiantes en riesgo. Pero ojo —esto es una herramienta de apoyo, no un oráculo. No reemplaza el criterio institucional ni el trabajo de tutores y docentes».

**Habla:** «Este fue mi proyecto final de Big Data. Los datos son del dataset público UCI Student Dropout, el código está en GitHub, y el dashboard interactivo —con el simulador— está en ObservableHQ para que lo prueben ustedes mismos».

*Sonreír, pausa.*

**Habla:** «Muchas gracias. ¿Preguntas?».

---

## ⏱ CHECKLIST DE TIEMPO

| Minuto | Sección | Acción clave |
|--------|---------|-------------|
| 0:00 | Hook | Pregunta al aula, mostrar portada |
| 0:35 | Stats + DAMA | 4 cards + gráfico calidad |
| 1:15 | EDA: balance | Barra 67.5/32.5 |
| 1:40 | EDA: selector | Cambiar variable, mostrar brecha |
| 2:15 | EDA: scatter | Círculo sobre zona roja |
| 2:45 | Treemap | 90.6% rendimiento |
| 3:45 | Modelos + métricas | Tabla, destacar RF |
| 4:40 | Matriz confusión | señalar FN y FP |
| 5:20 | Cards interpretación | Costos operativos |
| 5:55 | Demo 1: alto riesgo | 3 sliders → riesgo |
| 6:55 | Demo 2: bajo riesgo | 3 sliders → estable |
| 7:45 | Demo 3: intermedio | Caso límite |
| 9:00 | Conclusiones | 4 aprendizajes |
| 9:45 | Recomendación | Alertas tempranas |
| 10:05 | Limitaciones | 3 límites |
| 10:30 | Cierre | Pregunta final + gracias |

## 🗣 FRASES CLAVE PARA NO OLVIDAR

- *«El 90.6% de las deserciones son por bajo rendimiento»* — al mostrar treemap
- *«Preferimos 40 falsas alarmas antes que perder un estudiante sin intervención»* — al mostrar cards de costos
- *«El modelo corre en el navegador, sin servidores, sin internet»* — antes del simulador
- *«No reemplaza el criterio institucional»* — en el cierre

## 📊 NÚMEROS EXACTOS PARA REFERENCIA RÁPIDA

| Concepto | Valor |
|----------|-------|
| Total estudiantes | 4,424 |
| Variables | 36 |
| Desbalance | 32.48% deserción |
| DAMA | 100% (45 reglas) |
| Estables (0) | 2,987 (67.52%) |
| Deserción (1) | 1,437 (32.48%) |
| Treemap: Bajo Rend. | 90.6% |
| Treemap: Vulnerab. Fin. | 6.5% |
| Treemap: Desadaptación | 2.9% |
| Random Forest Recall | 95.82% |
| Random Forest Accuracy | 94.12% |
| RF F1 | 0.9136 |
| RF ROC-AUC | 0.9886 |
| Matriz: TN / FP | 558 / 40 |
| Matriz: FN / TP | 12 / 275 |
| Importancia: tasa aprob. | 54.8% |
| Importancia: estab. fin. | 15.0% |
| Importancia: edad | 11.6% |
