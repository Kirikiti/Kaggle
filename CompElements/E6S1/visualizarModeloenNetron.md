# 📌 Pasos para visualizar tu modelo en Netron

## 1. Activar tu entorno de Miniconda3
```bash
conda activate tu_entorno
```

## 2. Instalar las dependencias necesarias
Puedes instalar todo con pip:
```bash
pip install onnx onnxruntime netron
```

O, si quieres una instalación más estable:
```bash
conda install -c conda-forge onnx onnxruntime
pip install netron
```

---

## 3. Crear un archivo nuevo para exportar el modelo
Crea un archivo llamado:
```
export_model.py
```

Este archivo debe:
- Importar tu clase `MoERegressor`
- Instanciar el modelo con el `input_dim` correcto
- Cargar los pesos desde `moe_model_final.pth`
- Crear un tensor de entrada ficticio
- Exportar el modelo a ONNX o TorchScript

---

## 4. Ejecutar el script para generar el archivo ONNX
```bash
python export_model.py
```

Esto generará un archivo como:
```
moe_model.onnx
```

---

## 5. Abrir el modelo en Netron
Si instalaste Netron vía pip:
```bash
netron moe_model.onnx
```

Si instalaste la app de escritorio:
1. Abre Netron
2. Arrastra el archivo `moe_model.onnx` dentro de la ventana

---

## 6. Explorar la arquitectura del modelo
En Netron podrás ver:
- Las capas del modelo
- Los expertos del MoE
- El gating network
- Las dimensiones de entrada y salida
- La estructura completa del grafo
