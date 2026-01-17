# 🧩 Cambios entre el modelo MoE original y la versión compatible con ONNX

Este documento describe de forma clara y estructurada las diferencias entre:

- El modelo Mixture of Experts (MoE) original, basado en selección dinámica de expertos.
- La versión ONNX‑compatible, diseñada para permitir exportación a ONNX sin romper el entrenamiento.

La estructura sigue el formato solicitado:  
**Concepto cambiado → Antes → Ahora → Explicación.**

---

## 🔧 Concepto cambiado 1 — Selección dinámica de expertos

### ❌ Antes (modelo original)
```python
expert_out = torch.stack([
    self.experts[e](x[j].unsqueeze(0))
    for j, e in enumerate(expert_idx)
])
```

### ✔️ Ahora (modelo ONNX‑compatible)
```python
expert_outputs = []
for expert in self.experts:
    expert_outputs.append(expert(x))

expert_outputs = torch.stack(expert_outputs, dim=1)
```

### 📝 Explicación
El modelo original seleccionaba expertos según índices calculados en tiempo de ejecución.  
ONNX no soporta selección dinámica de módulos, bucles dependientes de datos ni list comprehensions con tensores.  
La versión ONNX ejecuta todos los expertos siempre, permitiendo un grafo estático exportable.

---

## 🔧 Concepto cambiado 2 — Routing Top‑K duro → mezcla suave

### ❌ Antes (modelo original)
```python
topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)
gate_scores = torch.softmax(topk_vals, dim=1)
```

### ✔️ Ahora (modelo ONNX‑compatible)
```python
gate_logits = self.w_gating(x)
gate_scores = torch.softmax(gate_logits, dim=1)
```

### 📝 Explicación
El modelo original usaba Top‑K routing, donde solo los mejores expertos participaban.  
ONNX no puede representar esta selección condicional.  
La versión ONNX usa una mezcla suave: todos los expertos reciben un peso, aunque muchos sean casi cero.

---

## 🔧 Concepto cambiado 3 — Bucle dependiente del batch → bucle fijo

### ❌ Antes (modelo original)
```python
for j, e in enumerate(expert_idx):
    ...
```

### ✔️ Ahora (modelo ONNX‑compatible)
```python
for expert in self.experts:
    expert_outputs.append(expert(x))
```

### 📝 Explicación
El bucle original dependía del contenido del batch, lo cual ONNX no puede convertir.  
El nuevo bucle es fijo y recorre únicamente los expertos, garantizando compatibilidad.

---

## 🔧 Concepto cambiado 4 — Combinación manual → combinación vectorizada

### ❌ Antes (modelo original)
```python
outputs.append(gate_scores[:, i].unsqueeze(1) * expert_out.squeeze(1))
return sum(outputs)
```

### ✔️ Ahora (modelo ONNX‑compatible)
```python
gate_scores = gate_scores.unsqueeze(-1)
output = torch.sum(expert_outputs * gate_scores, dim=1)
```

### 📝 Explicación
La combinación original usaba listas dinámicas y sumas iterativas.  
La versión ONNX usa operaciones vectorizadas, que ONNX puede representar sin problemas.

---

## 🔧 Concepto cambiado 5 — Eliminación de TopKGate

### ❌ Antes (modelo original)
```python
self.gate = TopKGate(input_dim, num_experts, k)
gate_scores, topk_idx = self.gate(x)
```

### ✔️ Ahora (modelo ONNX‑compatible)
```python
self.w_gating = nn.Linear(input_dim, num_experts)
gate_scores = torch.softmax(self.w_gating(x), dim=1)
```

### 📝 Explicación
TopKGate depende de torch.topk, que ONNX no puede convertir cuando se usa para seleccionar módulos.  
La versión ONNX usa un gating lineal estándar.

---

## 🔧 Concepto cambiado 6 — Arquitectura MoE simplificada pero equivalente

### ❌ Antes (modelo original)
- Routing duro (solo top‑k expertos)  
- Selección dinámica  
- Ejecución parcial de expertos  

### ✔️ Ahora (modelo ONNX‑compatible)
- Routing suave (todos los expertos)  
- Sin selección dinámica  
- Grafo estático exportable  

### 📝 Explicación
La versión ONNX mantiene la esencia del MoE, pero elimina toda operación dinámica que ONNX no puede convertir.  
El gating sigue controlando la importancia de cada experto, aunque ahora todos se evalúan siempre.

---

## 📌 Resumen final

| Componente | Original | ONNX‑compatible |
|-----------|----------|-----------------|
| Selección de expertos | Dinámica (top‑k) | Estática (todos los expertos) |
| Routing | Hard routing | Soft routing |
| Bucle | Dependiente de datos | Fijo |
| Exportable a ONNX | ❌ No | ✔️ Sí |
| Velocidad | Más rápido | Más lento |
| Calidad del modelo | Excelente | Muy similar |

---

## 🧠 Conclusión

La versión ONNX‑compatible mantiene la esencia del MoE, pero elimina toda operación dinámica que ONNX no puede convertir.  
El resultado es un modelo entrenable, estable, exportable y funcionalmente equivalente en la práctica.
