# Guía para construir tu notebook en ObservableHQ

**IMPORTANTE:** Cada celda define UNA sola variable. Sigue el orden exacto.

---

## 1. Archivos a subir (File Attachments)

Sube estos 14 archivos desde `proyecto_final/data/observable/`:

1. `metadata_proyecto.csv`
2. `calidad_dimensiones.csv`
3. `calidad_variables.csv`
4. `reglas_calidad_detalle.csv`
5. `eda_balance_clases.csv`
6. `eda_distribuciones.csv`
7. `eda_jerarquia_fallas.csv`
8. `metricas_modelos.csv`
9. `matriz_confusion.csv`
10. `importancia_variables.csv`
11. `predicciones_modelo.csv`
12. `modelo_final.onnx`
13. `modelo_web_config.json`
14. `eda_scatter.csv`

---

## 2. Orden de creación (importante)

**Paso 1:** Crea primero todas las celdas de la Sección 3 (soporte) — así cuando crees las celdas visuales, todas las variables ya existen.

**Paso 2:** Crea las celdas de la Sección 2 (visibles).

**Paso 3:** Arrastra las celdas de la Sección 3 al final del notebook, debajo de las conclusiones. Así queda limpio.

---

## 3. Celdas de soporte (CREAR PRIMERO, luego mover al final)

#### Celda S1 (JS — Tema de colores)
```javascript
theme = ({
  blue: "#1e40af",
  green: "#16a34a",
  amber: "#d97706",
  red: "#dc2626",
  slate: "#334155",
  muted: "#64748b",
  border: "#e2e8f0",
  bg: "#f8fafc"
})
```

#### Celda S2 (JS — D3)
```javascript
d3 = require("d3@7")
```

#### Celda S3 (JS — Plot)
```javascript
Plot = require("@observablehq/plot@0.6")
```

#### Celda S4 (JS — ONNX)
```javascript
ort = require("onnxruntime-web@1.16.0")
```

#### Celda S5 (JS — Archivo calidad_dimensiones)
```javascript
calidad_dimensiones = FileAttachment("calidad_dimensiones.csv").csv({typed: true})
```

#### Celda S6 (JS — Archivo reglas_calidad)
```javascript
reglas_calidad = FileAttachment("reglas_calidad_detalle.csv").csv({typed: true})
```

#### Celda S7 (JS — Archivo balance_clases)
```javascript
balance_clases = FileAttachment("eda_balance_clases.csv").csv({typed: true})
```

#### Celda S8 (JS — Archivo distribuciones_data)
```javascript
distribuciones_data = FileAttachment("eda_distribuciones.csv").csv({typed: true})
```

#### Celda S9 (JS — Archivo jerarquia_fallas)
```javascript
jerarquia_fallas = FileAttachment("eda_jerarquia_fallas.csv").csv({typed: true})
```

#### Celda S10 (JS — Archivo predicciones_modelo)
```javascript
predicciones_modelo = FileAttachment("predicciones_modelo.csv").csv({typed: true})
```

#### Celda S11 (JS — Archivo matriz_confusion)
```javascript
matriz_confusion = FileAttachment("matriz_confusion.csv").csv({typed: true})
```

#### Celda S12 (JS — Archivo ONNX)
```javascript
modelo_bytes = FileAttachment("modelo_final.onnx").arrayBuffer()
```

#### Celda S13 (JS — Sesión ONNX)
```javascript
session = {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.0/dist/";
  return ort.InferenceSession.create(modelo_bytes);
}
```

#### Celda S14 (JS — Función predecir)
```javascript
async function predecir(inputsArray) {
  const inputTensor = new ort.Tensor("float32", new Float32Array(inputsArray), [1, 9]);
  const feeds = { float_input: inputTensor };
  const results = await session.run(feeds);
  const outputLabel = results.label.data[0];
  const probabilities = results.probabilities.data;
  return {
    clase: outputLabel,
    probabilidad_desercion: probabilities[1]
  };
}
```

#### Celda S15 (JS — Datos scatter para EDA)
```javascript
scatter_data = FileAttachment("eda_scatter.csv").csv({typed: true})
```

#### Celda S16 (JS — Datos jerarquía para treemap)
```javascript
hierarchy_data = {
  const root = {name: "UNAJ", children: []};
  const by_type = d3.group(jerarquia_fallas, d => d.Type);
  for (const [type, items] of by_type) {
    const total_type = d3.sum(items, d => +d.cantidad);
    const type_node = {name: type, value: total_type, children: []};
    for (const item of items) {
      type_node.children.push({
        name: item.tipo_falla_especifico,
        value: +item.cantidad
      });
    }
    root.children.push(type_node);
  }
  return root;
}
```

---

## 4. Celdas visibles (CREAR DESPUÉS, quedan arriba)

### PORTADA

#### Celda 1 (JS — Header profesional)
```javascript
header = {
  const container = d3.create("section")
    .style("font-family", "system-ui, -apple-system, Segoe UI, sans-serif")
    .style("max-width", "1050px")
    .style("padding", "28px 0 10px")
    .style("line-height", "1.45");

  container.append("div")
    .style("font-size", "13px")
    .style("font-weight", "700")
    .style("letter-spacing", "0.06em")
    .style("text-transform", "uppercase")
    .style("color", theme.blue)
    .text("Trabajo Final | Big Data y Ciencia de Datos");

  container.append("h1")
    .style("font-size", "38px")
    .style("line-height", "1.05")
    .style("max-width", "920px")
    .style("margin", "12px 0")
    .text("¿Podemos detectar a tiempo qué estudiante va a abandonar la universidad?");

  container.append("p")
    .style("font-size", "17px")
    .style("max-width", "880px")
    .style("color", theme.slate)
    .text("Cada año, estudiantes dejan las aulas por razones que muchas veces podrían anticiparse. Este análisis explora si los datos del primer año —notas, situación financiera, edad— permiten identificar alumnos en riesgo y activar alertas antes de que sea tarde.");

  const cards = container.append("div")
    .style("display", "grid")
    .style("grid-template-columns", "repeat(3, minmax(0, 1fr))")
    .style("gap", "14px")
    .style("margin", "24px 0");

  const data = [
    {k: "Problema", v: "Deserción universitaria", d: "Detectar temprano qué alumnos están en riesgo de abandonar."},
    {k: "Datos", v: "4,424 estudiantes", d: "UCI Student Dropout — variables académicas, financieras y demográficas."},
    {k: "Objetivo", v: "Alerta temprana", d: "Clasificar: Estable (0) vs Deserción (1) usando Machine Learning."}
  ];

  const card = cards.selectAll("div")
    .data(data)
    .join("div")
    .style("border", `1px solid ${theme.border}`)
    .style("border-radius", "8px")
    .style("padding", "16px")
    .style("background", theme.bg);

  card.append("div")
    .style("font-size", "12px")
    .style("font-weight", "700")
    .style("color", theme.muted)
    .text(d => d.k);

  card.append("div")
    .style("font-size", "22px")
    .style("font-weight", "800")
    .style("margin", "6px 0")
    .text(d => d.v);

  card.append("div")
    .style("font-size", "13px")
    .style("color", theme.slate)
    .text(d => d.d);

  container.append("blockquote")
    .style("border-left", `5px solid ${theme.blue}`)
    .style("background", "#eff6ff")
    .style("padding", "14px 18px")
    .style("margin", "20px 0 0")
    .text("Pregunta: ¿En qué medida las variables socioeconómicas y el rendimiento del primer año permiten predecir qué estudiantes desertarán?");

  return container.node();
}
```

#### Celda 2 (JS — Números clave)
```javascript
header_stats = {
  const container = d3.create("section")
    .style("font-family", "system-ui, -apple-system, Segoe UI, sans-serif")
    .style("max-width", "1050px")
    .style("line-height", "1.5");

  const cards = container.append("div")
    .style("display", "grid")
    .style("grid-template-columns", "repeat(4, minmax(0, 1fr))")
    .style("gap", "14px")
    .style("margin", "16px 0");

  const cardData = [
    {titulo: "Variables", valor: "36", detalle: "Originales + ingeniería"},
    {titulo: "Desbalance", valor: "32.4%", detalle: "Tasa de deserción real"},
    {titulo: "Calidad DAMA", valor: "100%", detalle: "Cumplimiento en datos"},
    {titulo: "Predicción", valor: "ONNX", detalle: "En navegador, tiempo real"}
  ];

  const card = cards.selectAll("div")
    .data(cardData)
    .join("div")
    .style("border", `1px solid ${theme.border}`)
    .style("border-radius", "8px")
    .style("padding", "14px")
    .style("background", theme.bg)
    .style("text-align", "center");

  card.append("div")
    .style("font-size", "11px")
    .style("font-weight", "700")
    .style("color", theme.muted)
    .style("text-transform", "uppercase")
    .style("letter-spacing", "0.05em")
    .text(d => d.titulo);

  card.append("div")
    .style("font-size", "28px")
    .style("font-weight", "800")
    .style("margin", "6px 0")
    .style("color", theme.blue)
    .text(d => d.valor);

  card.append("div")
    .style("font-size", "12px")
    .style("color", theme.slate)
    .text(d => d.detalle);

  return container.node();
}
```

---

### CALIDAD DE DATOS

#### Celda 3 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h2 style="margin-top:28px">1. Los datos son confiables</h2>
</section>`
```

#### Celda 4 (JS — Gráfico DAMA)
```javascript
Plot.plot({
  title: "Cumplimiento por dimensión DAMA",
  subtitle: "100% en todas las dimensiones",
  x: {label: "Cumplimiento (%)", domain: [0, 100]},
  y: {label: ""},
  marginLeft: 120,
  color: {legend: true, range: ["#1e40af", "#16a34a", "#d97706", "#dc2626"]},
  marks: [
    Plot.barX(calidad_dimensiones, {x: "cumplimiento_pct", y: "dimension", fill: "dimension", tip: true}),
    Plot.ruleX([100])
  ]
})
```

#### Celda 5 (JS — Tabla de reglas)
```javascript
Inputs.table(reglas_calidad, {
  columns: ["dimension", "regla", "estado", "registros_evaluados", "registros_incumplen"],
  header: {
    dimension: "Dimensión",
    regla: "Regla",
    estado: "Estado",
    registros_evaluados: "Evaluados",
    registros_incumplen: "Incumplen"
  }
})
```

---

### BALANCE DE CLASES

#### Celda 6 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h2 style="margin-top:28px">2. El problema del desbalance</h2>
</section>`
```

#### Celda 7 (JS — Gráfico de balance)
```javascript
Plot.plot({
  title: "Distribución: Estables vs Deserción",
  x: {label: "", tickFormat: d => d === "0" ? "Estable" : "Deserción"},
  y: {label: "Porcentaje (%)", domain: [0, 100]},
  color: {range: [theme.blue, theme.red]},
  marks: [
    Plot.barY(balance_clases, {x: "Machine_failure", y: "porcentaje", fill: "Machine_failure", tip: true}),
    Plot.text(balance_clases, {x: "Machine_failure", y: "porcentaje", text: d => `${d.porcentaje}%`, dy: -8, fill: "currentColor"})
  ]
})
```

---

### EXPLORACIÓN DE DATOS

#### Celda 8 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h2 style="margin-top:28px">3. ¿Qué diferencia a quienes desertan?</h2>
</section>`
```

#### Celda 9 (JS — Selector)
```javascript
viewof selected_var = Inputs.select([
  {label: "Edad de matrícula", value: "Age at enrollment"},
  {label: "Notas 1er semestre", value: "Curricular units 1st sem (grade)"},
  {label: "Notas 2do semestre", value: "Curricular units 2nd sem (grade)"},
  {label: "Tasa de aprobación anual", value: "tasa_aprobacion_ano"},
  {label: "Estabilidad financiera", value: "estabilidad_financiera"}
], {
  format: d => d.label,
  label: "Variable"
})
```

#### Celda 10 (JS — Gráfico comparativo)
```javascript
Plot.plot({
  title: `Promedio de ${selected_var.label}`,
  x: {label: "Estable (0) vs Deserción (1)"},
  y: {label: `Promedio de ${selected_var.label}`},
  color: {range: [theme.blue, theme.red]},
  marks: [
    Plot.barY(distribuciones_data, {
      x: "Machine failure",
      y: `${selected_var.value}_mean`,
      fill: "Machine failure",
      fx: "Type",
      tip: true
    })
  ]
})
```

#### Celda 10b (JS — Scatterplot interactivo)
```javascript
Plot.plot({
  title: "Edad vs Tasa de Aprobación — ¿Dónde se concentra la deserción?",
  subtitle: "Cada hexágono agrupa múltiples estudiantes. Color = proporción de deserción en esa zona.",
  x: {label: "Edad al matricularse", tickFormat: d => d},
  y: {label: "Tasa de aprobación anual", domain: [0, 1]},
  color: {legend: true, domain: [0, 1], range: [theme.blue, theme.red], tickFormat: d => d === 0 ? "Estable (0)" : "Deserción (1)"},
  marks: [
    Plot.hexbin(scatter_data, {x: "Age at enrollment", y: "tasa_aprobacion_ano", fill: "Machine failure", reduce: "mean", thresholds: 80, tip: true}),
    Plot.ruleY([0])
  ]
})
```

#### Celda 11 (JS — Insight)
```javascript
html`<blockquote style="border-left:5px solid ${theme.blue};background:#eff6ff;padding:12px 16px;margin:16px 0;max-width:1050px;font-family:system-ui,sans-serif;font-size:15px;color:#1e3a5f">
  <strong>¿Qué vemos?</strong> En promedio, los estudiantes que desertan tienen <strong>menor tasa de aprobación</strong>, 
  <strong>menor estabilidad financiera</strong> y son <strong>mayores</strong> al ingresar. Esto se repite tanto en jornada diurna como vespertina.
</blockquote>`
```

---

### JERARQUÍA DE DESERCIONES

#### Celda 12 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h3 style="margin-top:28px">¿Por qué desertan?</h3>
</section>`
```

#### Celda 13 (JS — Treemap interactivo)
```javascript
treemapChart = {
  const width = 850;
  const height = 360;

  const root = d3.hierarchy(hierarchy_data)
    .sum(d => d.value)
    .sort((a, b) => b.value - a.value);

  d3.treemap()
    .size([width, height])
    .paddingTop(22)
    .paddingInner(3)
    .paddingOuter(2)(root);

  const total = d3.sum(root.leaves(), d => d.value);

  const colorMotivo = d3.scaleOrdinal()
    .domain(["Bajo Rendimiento Académico", "Vulnerabilidad Financiera", "Desadaptación/Otros"])
    .range(["#dc2626", "#d97706", "#6b7280"]);

  const svg = d3.create("svg")
    .attr("viewBox", [0, 0, width, height])
    .attr("style", "max-width: 100%; height: auto; font-family: system-ui, sans-serif;");

  const group = svg.selectAll("g")
    .data(root.descendants().filter(d => d.depth === 1))
    .join("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

  group.append("rect")
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr("fill", d => colorMotivo(d.data.name))
    .attr("opacity", 0.08)
    .attr("rx", 6);

  group.append("text")
    .attr("x", 8)
    .attr("y", 16)
    .attr("font-size", 12)
    .attr("font-weight", "700")
    .attr("fill", d => colorMotivo(d.data.name))
    .text(d => d.data.name);

  const leaf = svg.selectAll("g.leaf")
    .data(root.leaves())
    .join("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`)
    .style("cursor", "pointer")
    .on("mouseenter", function() { d3.select(this).select("rect").transition(80).attr("opacity", 1); })
    .on("mouseleave", function() { d3.select(this).select("rect").transition(80).attr("opacity", 0.85); });

  leaf.append("rect")
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr("fill", d => colorMotivo(d.data.name))
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .attr("opacity", 0.85)
    .attr("rx", 4);

  leaf.append("title")
    .text(d => `${d.parent.data.name} — ${d.data.name}\n${d.value.toLocaleString()} alumnos (${(d.value / total * 100).toFixed(1)}%)`);

  leaf.append("text")
    .attr("x", 6)
    .attr("y", 16)
    .attr("font-size", 11)
    .attr("font-weight", "700")
    .attr("fill", "white")
    .attr("pointer-events", "none")
    .text(d => (d.x1 - d.x0 > 60 && d.y1 - d.y0 > 24) ? d.data.name : "");

  leaf.append("text")
    .attr("x", 6)
    .attr("y", 30)
    .attr("font-size", 10)
    .attr("fill", "rgba(255,255,255,0.9)")
    .attr("pointer-events", "none")
    .text(d => (d.x1 - d.x0 > 60 && d.y1 - d.y0 > 38) ? `${d.value.toLocaleString()} alumnos` : "");

  leaf.append("text")
    .attr("x", 6)
    .attr("y", 42)
    .attr("font-size", 9)
    .attr("fill", "rgba(255,255,255,0.7)")
    .attr("pointer-events", "none")
    .text(d => (d.x1 - d.x0 > 60 && d.y1 - d.y0 > 52) ? `${(d.value / total * 100).toFixed(1)}% del total` : "");

  return svg.node();
}
```

---

### DESEMPEÑO DEL MODELO

#### Celda 14 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h2 style="margin-top:28px">4. ¿Qué tan bien detecta el riesgo?</h2>
</section>`
```

#### Celda 15 (JS — Heatmap matriz de confusión)
```javascript
{
  const width = 400;
  const height = 340;
  const margin = {top: 40, right: 20, bottom: 60, left: 70};

  const container = d3.create("section")
    .style("font-family", "system-ui, sans-serif")
    .style("max-width", "500px");

  const svg = container.append("svg")
    .attr("viewBox", [0, 0, width, height])
    .style("width", "100%")
    .style("height", "auto");

  const clases = ["0 (Estable)", "1 (Deserción)"];
  const x = d3.scaleBand()
    .domain(clases)
    .range([margin.left, width - margin.right])
    .padding(0.1);

  const y = d3.scaleBand()
    .domain(clases)
    .range([margin.top, height - margin.bottom])
    .padding(0.1);

  const color = d3.scaleSequential()
    .domain([0, d3.max(matriz_confusion, d => d.cantidad)])
    .interpolator(d3.interpolateReds);

  svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).tickFormat(d => d.includes("1") ? "Pred.\nDeserción" : "Pred.\nEstable"));

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).tickFormat(d => d.includes("1") ? "Real\nDeserción" : "Real\nEstable"));

  svg.selectAll("rect")
    .data(matriz_confusion)
    .join("rect")
    .attr("x", d => x(d.prediccion))
    .attr("y", d => y(d.real))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("fill", d => color(d.cantidad))
    .attr("stroke", "#fff")
    .attr("stroke-width", 2);

  svg.selectAll("text.cell")
    .data(matriz_confusion)
    .join("text")
    .attr("x", d => x(d.prediccion) + x.bandwidth() / 2)
    .attr("y", d => y(d.real) + y.bandwidth() / 2)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("font-size", 20)
    .attr("font-weight", "700")
    .attr("fill", d => d.cantidad > d3.max(matriz_confusion, x => x.cantidad) * 0.55 ? "white" : "#111827")
    .text(d => d.cantidad);

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 14)
    .attr("text-anchor", "middle")
    .attr("font-weight", "700")
    .attr("font-size", 13)
    .text("Clase predicha");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", 22)
    .attr("text-anchor", "middle")
    .attr("font-weight", "700")
    .attr("font-size", 13)
    .text("Clase real");

  return container.node();
}
```

---

### SIMULADOR

#### Celda 16 (JS — Título)
```javascript
html`<section style="max-width:1050px;font-family:system-ui,sans-serif">
  <h2 style="margin-top:28px">5. Pruébalo tú mismo</h2>
</section>`
```

#### Celda 17 (JS — Tabla)
```javascript
viewof selected_student = Inputs.table(predicciones_modelo, {
  multiple: false,
  columns: ["id_maquina", "Age at enrollment", "tasa_aprobacion_ano", "estabilidad_financiera"],
  header: {
    id_maquina: "ID",
    "Age at enrollment": "Edad",
    tasa_aprobacion_ano: "Tasa Aprob.",
    estabilidad_financiera: "Estab. Fin."
  }
})
```

#### Celda 18 (JS — Sincronización)
```javascript
alumno_seleccionado = selected_student && !Array.isArray(selected_student)
  ? selected_student
  : (selected_student && selected_student.length > 0 ? selected_student[0] : null)
```

#### Celda 19 (JS — Slider edad)
```javascript
viewof edad_matricula = Inputs.range([17, 60], {
  value: alumno_seleccionado ? alumno_seleccionado["Age at enrollment"] : 18,
  step: 1,
  label: "Edad de matrícula"
})
```

#### Celda 20 (JS — Slider tasa aprobación)
```javascript
viewof tasa_aprobacion = Inputs.range([0, 100], {
  value: alumno_seleccionado ? alumno_seleccionado.tasa_aprobacion_ano * 100 : 80,
  step: 1,
  label: "Tasa de aprobación (%)"
})
```

#### Celda 21 (JS — Slider estabilidad)
```javascript
viewof estabilidad_financiera = Inputs.range([-5, 5], {
  value: alumno_seleccionado ? alumno_seleccionado.estabilidad_financiera : 0,
  step: 0.1,
  label: "Estabilidad financiera"
})
```

#### Celda 22 (JS — Inferencia reactiva)
```javascript
resultado_simulacion = {
  if (typeof edad_matricula === 'undefined' || typeof tasa_aprobacion === 'undefined' || typeof estabilidad_financiera === 'undefined') {
    return { clase: 0, probabilidad_desercion: 0.0 };
  }
  const inputs = [1, edad_matricula, 1, 1, 0, 1, tasa_aprobacion / 100, 0.0, estabilidad_financiera];
  return predecir(inputs);
}
```

#### Celda 23 (JS — Resultado)
```javascript
html`<div style="
  padding: 24px;
  border-radius: 10px;
  background-color: ${resultado_simulacion.clase === 1 ? '#fef2f2' : '#f0fdf4'};
  border: 2px solid ${resultado_simulacion.clase === 1 ? '#ef4444' : '#22c55e'};
  color: ${resultado_simulacion.clase === 1 ? '#991b1b' : '#166534'};
  font-family: system-ui, sans-serif;
  max-width: 450px;
  margin-top: 12px">
  <div style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">
    Resultado
  </div>
  <div style="font-size:24px;font-weight:800;margin:8px 0">
    ${resultado_simulacion.clase === 1 ? '⚠ Riesgo de deserción' : '✅ Estudiante estable'}
  </div>
  <div style="font-size:16px">
    Probabilidad de deserción: <strong>${(resultado_simulacion.probabilidad_desercion * 100).toFixed(2)}%</strong>
  </div>
</div>`
```

---

### CONCLUSIONES

#### Celda 24 (JS — Conclusiones)
```javascript
{
  const container = d3.create("section")
    .style("font-family", "system-ui, -apple-system, Segoe UI, sans-serif")
    .style("max-width", "1050px")
    .style("line-height", "1.5")
    .style("margin-top", "32px");

  container.append("h2").text("¿Qué aprendimos?");

  const bullets = [
    "La tasa de aprobación del primer año es la variable más determinante. Un alumno que reprueba materias desde el inicio tiene alta probabilidad de desertar.",
    "La estabilidad financiera y la edad de matrícula también pesan. No es solo un tema académico, también económico.",
    "El 90.6% de las deserciones se concentran en Bajo Rendimiento Académico, lo que confirma que el desempeño inicial es clave.",
    "El simulador ONNX permite ejecutar Machine Learning directamente en el navegador, sin servidores ni conexión a internet."
  ];

  const list = container.append("ul")
    .style("font-size", "16px");

  list.selectAll("li")
    .data(bullets)
    .join("li")
    .style("margin", "10px 0")
    .style("color", theme.slate)
    .text(d => d);

  container.append("h3")
    .style("margin-top", "24px")
    .style("font-size", "18px")
    .text("Recomendación");

  container.append("p")
    .style("font-size", "15px")
    .style("color", theme.slate)
    .style("max-width", "880px")
    .text("Implementar un sistema de alertas tempranas que monitoree la tasa de aprobación del primer semestre. Los estudiantes con menos del 50% de materias aprobadas deberían recibir tutoría académica y evaluación financiera antes del segundo semestre.");

  container.append("h3")
    .style("margin-top", "24px")
    .style("font-size", "18px")
    .text("Limitaciones");

  const limitaciones = [
    "Los datos provienen de una universidad europea (UCI Dataset), por lo que los patrones pueden no replicarse exactamente en el contexto argentino.",
    "El modelo usa datos del primer año. No considera factores externos como cambios de política educativa o crisis económicas durante la carrera.",
    "La variable objetivo es binaria (deserta/no deserta). No distingue entre abandono temporal, cambio de carrera o graduación exitosa."
  ];

  const limList = container.append("ul")
    .style("font-size", "14px")
    .style("color", theme.muted);

  limList.selectAll("li")
    .data(limitaciones)
    .join("li")
    .style("margin", "6px 0")
    .text(d => d);

  container.append("blockquote")
    .style("border-left", `5px solid ${theme.blue}`)
    .style("background", "#eff6ff")
    .style("padding", "14px 18px")
    .style("margin-top", "22px")
    .style("font-size", "15px")
    .style("color", "#1e3a5f")
    .text("Demo académica con datos del dataset UCI Student Dropout. El modelo es una herramienta de apoyo, no reemplaza el criterio institucional.");

  return container.node();
}
```


