import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# UTILIDADES
# ============================================================

def log(msg):
    """Escribe mensajes en training_log.txt en vez de print."""
    with open("training_log.txt", "a") as f:
        f.write(msg + "\n")
    
    """Tambein lo muestra en consola"""
    print(msg)

# ============================================================
# DATA PRE-PROCESSOR
# ============================================================

class TabularPreprocessor:
    def __init__(self, target_col):
        self.target_col = target_col
        self.categorical_cols = None
        self.feature_cols = None
        self.te = None
        self.scaler = None

    def fit(self, df):
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.feature_cols = [c for c in df.columns if c != self.target_col]

        df_t = df.copy()

        # Target Encoder
        self.te = TargetEncoder(cols=self.categorical_cols)
        df_t[self.categorical_cols] = self.te.fit_transform(
            df[self.categorical_cols], df[self.target_col]
        )

        # Standard Scaler
        self.scaler = StandardScaler()
        df_t[self.feature_cols] = self.scaler.fit_transform(df_t[self.feature_cols])

        return df_t

    def transform(self, df):
        df_t = df.copy()
        df_t[self.categorical_cols] = self.te.transform(df_t[self.categorical_cols])
        df_t[self.feature_cols] = self.scaler.transform(df_t[self.feature_cols])
        return df_t

    def transform_to_arrays(self, df):
        df_t = self.transform(df)
        X = df_t.drop(columns=[self.target_col]).values.astype(np.float32)
        y = df_t[self.target_col].values.astype(np.float32)
        return X, y

# ============================================================
# DATASET
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MIXTURE OF EXPERTS (MoE)
# ============================================================

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class TopKGate(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.k = k
        self.w_gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.w_gating(x)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)
        gate_scores = torch.softmax(topk_vals, dim=1)
        return gate_scores, topk_idx

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.w_gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: [B, input_dim]

        # 1) Gating: [B, num_experts]
        gate_logits = self.w_gating(x)
        gate_scores = torch.softmax(gate_logits, dim=1)

        # 2) Ejecutar todos los expertos: lista de [B, hidden_dim]
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # 3) Apilar: [B, num_experts, hidden_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 4) Expandir gating: [B, num_experts, 1]
        gate_scores = gate_scores.unsqueeze(-1)

        # 5) Combinación ponderada: [B, hidden_dim]
        output = torch.sum(expert_outputs * gate_scores, dim=1)

        return output


class MoERegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_experts=4):
        super().__init__()
        self.moe = MoELayer(input_dim, hidden_dim, num_experts)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, input_dim]
        moe_out = self.moe(x)          # [B, hidden_dim]
        out = self.out(moe_out)        # [B, 1]
        return out

# ============================================================
# ENTRENAMIENTO
# ============================================================

def fit_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss = float("inf")
    patience = 7
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDACIÓN
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        log(f"Epoch {epoch+1} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}")

        # EARLY STOPPING
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log("Early stopping activado")
                break

    return best_loss

# ============================================================
# MAIN
# ============================================================

def main():
    # Limpia el log anterior
    open("training_log.txt", "w").close()

    log("Cargando datos...")
    train_df = pd.read_csv("Descargas/train.csv")
    test_df = pd.read_csv("Descargas/test.csv")
    sample = pd.read_csv("Descargas/sample_submission.csv")

    target_col = "exam_score"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Usando dispositivo: {device}")

    # ============================================================
    # K-FOLD
    # ============================================================

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        log(f"\n===== FOLD {fold+1} =====")

        train_data = train_df.iloc[train_idx].copy()
        val_data = train_df.iloc[val_idx].copy()

        # Preprocesador por fold
        pre = TabularPreprocessor(target_col)

        train_processed = pre.fit(train_data)
        val_processed = pre.transform(val_data)

        # Convertir a arrays
        X_train = train_processed.drop(columns=[target_col]).values.astype(np.float32)
        y_train = train_processed[target_col].values.astype(np.float32)

        X_val = val_processed.drop(columns=[target_col]).values.astype(np.float32)
        y_val = val_processed[target_col].values.astype(np.float32)

        # Datasets
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Modelo con input_dim correcto
        model = MoERegressor(input_dim=len(pre.feature_cols)).to(device)

        val_loss = fit_model(model, train_loader, val_loader, device)
        fold_losses.append(val_loss)

        log(f"Fold {fold+1} Val Loss: {val_loss:.5f}")

        # Guardar modelo k-fold
        torch.save(model.state_dict(), f"moe_fold_{fold+1}.pth")
        log(f"Modelo fold_{fold+1} guardado como moe_fold_{fold+1}.pth")

    # Guarda pérdidas de los folds
    with open("fold_losses.json", "w") as f:
        json.dump(fold_losses, f, indent=4)

    # ============================================================
    # ENTRENAMIENTO FINAL CON TODO train_df
    # ============================================================

    log("\nEntrenando modelo final con todo el dataset...")

    pre_final = TabularPreprocessor(target_col)
    train_processed = pre_final.fit(train_df)

    X_full = train_processed.drop(columns=[target_col]).values.astype(np.float32)
    y_full = train_processed[target_col].values.astype(np.float32)

    full_dataset = TabularDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

    final_model = MoERegressor(input_dim=len(pre_final.feature_cols)).to(device)
    
    # Utilizamos metodo fit_model
    final_loss = fit_model(final_model, full_loader, full_loader, device)

    with open("final_train_loss.txt", "w") as f:
        f.write(str(final_loss))

    log(f"Final Train Loss: {final_loss:.5f}")

    # Guardar modelo final
    torch.save(final_model.state_dict(), "moe_model_final.pth")
    log("Modelo final guardado como moe_model_final.pth")

    # ============================================================
    # PREDICCIÓN SOBRE TEST
    # ============================================================

    log("\nGenerando predicciones para test...")

    # Transformar test_df con el preprocesador final
    test_processed = pre_final.transform(test_df)

    # Convertir a tensores
    X_test = test_processed.values.astype(np.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    final_model.eval()
    with torch.no_grad():
        preds = final_model(test_tensor).cpu().numpy().flatten()

    sample[target_col] = preds
    sample.to_csv("submission.csv", index=False)

    log("Archivo submission.csv generado")
    log("\nEntrenamiento completado.")

if __name__ == "__main__":
    main()
