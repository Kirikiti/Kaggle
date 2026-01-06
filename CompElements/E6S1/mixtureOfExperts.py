import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
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

# ============================================================
# DATASET
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, df, target_col):
        self.X = df.drop(columns=[target_col]).values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)

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
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = TopKGate(input_dim, num_experts, k)

    def forward(self, x):
        gate_scores, topk_idx = self.gate(x)
        outputs = []
        for i in range(gate_scores.size(1)):
            expert_idx = topk_idx[:, i]
            expert_out = torch.stack([self.experts[e](x[j].unsqueeze(0)) for j, e in enumerate(expert_idx)])
            outputs.append(gate_scores[:, i].unsqueeze(1) * expert_out.squeeze(1))
        return sum(outputs)

class MoERegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_experts=4, k=2):
        super().__init__()
        self.moe = MoELayer(input_dim, hidden_dim, num_experts, k)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out(self.moe(x))

# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_one_fold(model, train_loader, val_loader, device):
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
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    target_col = "target"
    input_dim = train_df.drop(columns=[target_col]).shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Usando dispositivo: {device}")

    # ============================================================
    # K-FOLD
    # ============================================================

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        log(f"\n===== FOLD {fold+1} =====")

        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]

        train_loader = DataLoader(TabularDataset(train_data, target_col), batch_size=256, shuffle=True)
        val_loader = DataLoader(TabularDataset(val_data, target_col), batch_size=256, shuffle=False)

        model = MoERegressor(input_dim).to(device)
        val_loss = train_one_fold(model, train_loader, val_loader, device)
        fold_losses.append(val_loss)

        log(f"Fold {fold+1} Val Loss: {val_loss:.5f}")

    # Guarda pérdidas de los folds
    with open("fold_losses.json", "w") as f:
        json.dump(fold_losses, f, indent=4)

    # ============================================================
    # ENTRENAMIENTO FINAL CON TODO train_df
    # ============================================================

    log("\nEntrenando modelo final con todo el dataset...")
    full_loader = DataLoader(TabularDataset(train_df, target_col), batch_size=256, shuffle=True)

    final_model = MoERegressor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=1e-3)

    final_loss = 0.0
    final_model.train()
    for X, y in full_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = final_model(X)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
        final_loss += loss.item()

    final_loss /= len(full_loader)

    with open("final_train_loss.txt", "w") as f:
        f.write(str(final_loss))

    log(f"Final Train Loss: {final_loss:.5f}")

    # ============================================================
    # PREDICCIONES Y GUARDADO DEL MODELO
    # ============================================================

    torch.save(final_model.state_dict(), "moe_model_final.pth")
    log("Modelo final guardado como moe_model_final.pth")

    final_model.eval()
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32).to(device)
    preds = final_model(test_tensor).detach().cpu().numpy().flatten()

    sample["target"] = preds
    sample.to_csv("submission.csv", index=False)
    log("Archivo submission.csv generado")

    log("\nEntrenamiento completado.")

if __name__ == "__main__":
    main()
