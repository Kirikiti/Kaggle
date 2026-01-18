import torch
import pandas as pd
from moe_e6s1_ONNX import MoERegressor                 # Ajusta el nombre del archivo donde está tu clase
from moe_e6s1_ONNX import TabularPreprocessor  # Ajusta el nombre del archivo donde está tu preprocesador

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "moe_model_final_fONNX.pth"
TRAIN_PATH = "/home/pi/Descargas/train.csv"
TARGET_COL = "exam_score"
ONNX_OUTPUT = "moe_model.onnx"

device = torch.device("cpu")   # Exportar siempre en CPU

# ============================================================
# 1. CARGAR DATASET PARA RECONSTRUIR input_dim
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

# Preprocesar igual que en entrenamiento final
pre = TabularPreprocessor(TARGET_COL)
train_processed = pre.fit(train_df)

input_dim = len(pre.feature_cols)

# ============================================================
# 2. RECONSTRUIR EL MODELO Y CARGAR PESOS
# ============================================================

model = MoERegressor(input_dim=input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ============================================================
# 3. CREAR TENSOR DE ENTRADA FICTICIO
# ============================================================

dummy_input = torch.randn(1, input_dim, dtype=torch.float32).to(device)

# ============================================================
# 4. EXPORTAR A ONNX
# ============================================================

torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"Modelo exportado correctamente a {ONNX_OUTPUT}")
