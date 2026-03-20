"""
hvac_energy_forecasting.py

This script implements deep learning models (LSTM, BiLSTM, GRU, SSM-inspired)
to forecast HVAC chiller energy consumption using hourly time-series data.

 It includes:
- Data loading and preprocessing
- Feature engineering (temporal, lag, rolling statistics)
-Sequencecreation for LSTM/GRU models
- Model definitions
- Training with early stoping and learning rate scheduler
- Evaluation metrics (RMSE, MAE, R², MAPE, Accuracy)
- Baseline models(Random Forest, XGBoost)
- Ensemble predictions and visualization

Authors: Abrha Dawit Nigusse
 
"""

# ================================================================
#  Chiller Forcasting- LSTM / BiLSTM/ GRU / CPU-ONLY SSM model
#  (Zero external deps – pure PyTorch )

# ================================================================
import os, random, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ------------------- Reproducibility -------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ------------------- Data & Features (unchanged) -------------------
def load_hvac_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Local Time (Timezone : GMT+8h)"])
    df.rename(columns={"Local Time (Timezone : GMT+8h)": "timestamp"}, inplace=True)
    df.sort_values("timestamp", inplace=True); df.reset_index(drop=True, inplace=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df.ffill(inplace=True); df.bfill(inplace=True)
    return df

def add_temporal_features(df, target):
    base = ["Cooling Water Temperature (C)","Humidity (%)","Building Load (RT)",
            "Chilled Water Rate (L/sec)","Outside Temperature (F)","Dew Point (F)",
            "Wind Speed (mph)","Pressure (in)","hour","day_of_week"]
    feats = base[:]
    for lag in [1,2,3]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
        feats.append(f"{target}_lag{lag}")
    for w in [3,6]:
        df[f"{target}_roll_mean_{w}"] = df[target].rolling(w).mean()
        df[f"{target}_roll_std_{w}"]  = df[target].rolling(w).std()
        feats.extend([f"{target}_roll_mean_{w}", f"{target}_roll_std_{w}"])
    df.ffill(inplace=True); df.bfill(inplace=True)
    df[target] = df[target].clip(lower=df[target].quantile(0.01),
                                 upper=df[target].quantile(0.99))
    return df, feats

def create_sequences(X, y, look_back=72):
    Xs, ys = [], []
    for i in range(look_back, len(X)):
        Xs.append(X[i-look_back:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def scale_splits(train_df, val_df, test_df, feats, target):
    f_scaler = MinMaxScaler().fit(train_df[feats])
    t_scaler = MinMaxScaler().fit(train_df[[target]])
    return (f_scaler.transform(train_df[feats]), f_scaler.transform(val_df[feats]),
            f_scaler.transform(test_df[feats]),
            t_scaler.transform(train_df[[target]]), t_scaler.transform(val_df[[target]]),
            t_scaler.transform(test_df[[target]]), f_scaler, t_scaler)

# ------------------- RNN models (unchanged) -------------------
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True, dropout=0, num_layers=1)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True, dropout=0.0, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32); self.fc2 = nn.Linear(32, 1); self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm1(x); out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out)); return self.fc2(out)

class BiLSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden*2); self.dropout1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(hidden*2, hidden//2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, 32); self.fc2 = nn.Linear(32, 1); self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.bilstm1(x); out = self.bn(out[:, -1, :]); out = self.dropout1(out)
        out, _ = self.bilstm2(out.unsqueeze(1)); out = self.dropout2(out[:, -1, :])
        out = self.relu(self.fc1(out)); return self.fc2(out)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden, batch_first=True, dropout=0.2, num_layers=1)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True, dropout=0.0, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32); self.fc2 = nn.Linear(32, 1); self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.gru1(x); out, _ = self.gru2(out)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out)); return self.fc2(out)

# ------------------- CPU-ONLY SSM (no external libs) -------------------
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        # diagonal A for stability
        self.A_log = nn.Parameter(torch.zeros(d_state))
        # project input -> (3 * d_state)
        self.proj = nn.Linear(d_model, 3*d_state)
        self.D = nn.Parameter(torch.ones(d_model))  # output residual

    def forward(self, x):
        B, L, D = x.shape
        # project
        params = self.proj(x)               # (B, L, 3*d_state)
        dt, Bp, Cp = params.chunk(3, dim=-1)  # each (B, L, d_state)
        dt = F.softplus(dt)                 # ensure positive
        A = -torch.exp(self.A_log)          # stable negative diagonal

        h = torch.zeros(B, self.d_state, device=x.device)
        outs = []
        for t in range(L):
            # elementwise ops: all (B, d_state)
            h = A * h + dt[:, t] * Bp[:, t]   # (B, d_state)
            y_t = (Cp[:, t] * h).sum(dim=-1) + (x[:, t] * self.D).sum(dim=-1)  # (B,)
            outs.append(y_t.unsqueeze(1))
        return torch.cat(outs, dim=1)  # (B, L)

class SSMNet(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16, n_layers=3, dropout=0.3):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        self.blocks  = nn.ModuleList([SSMBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.proj_in(x)        # (B, L, d_model)
        for blk in self.blocks:
            y = blk(x)             # (B, L)
            # expand last dim to match d_model for residual
            y_exp = y.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x = self.norm(x + y_exp)
        x = self.dropout(x[:, -1, :])
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ------------------- Training (CPU-friendly) -------------------
def train_model(model, train_loader, val_loader,
                epochs=120, lr=1e-3, patience_es=12, patience_lr=4):
    device = torch.device('cpu')                     # <-- force CPU
    model.to(device); criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                factor=0.5, patience=patience_lr, threshold=1e-4, min_lr=1e-6)

    best, wait = float('inf'), 0
    tr_l, va_l = [], []
    for epoch in range(1, epochs+1):
        # ---- train ----
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(train_loader.dataset); tr_l.append(tr_loss)

        # ---- valid ----
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()*xb.size(0)
        va_loss /= len(val_loader.dataset); va_l.append(va_loss)
        sched.step(va_loss)

        if va_loss < best:
            best, wait = va_loss, 0
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        else:
            wait += 1
            if wait >= patience_es:
                print(f"   Early-stop @ epoch {epoch}")
                break

        if epoch % 20 == 0 or epoch <= 3:
            print(f"Epoch {epoch:3d} | Train {tr_loss:.6f} | Val {va_loss:.6f}")

    model.load_state_dict(torch.load(f"{model.__class__.__name__}_best.pth"))
    return model, tr_l, va_l

# ------------------- Evaluation -------------------
@torch.no_grad()
def evaluate_model(model, loader, t_scaler):
    device = torch.device('cpu')
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
    return (t_scaler.inverse_transform(y_true),
            t_scaler.inverse_transform(y_pred))

def metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/np.clip(y_true,1e-8,None)))*100
    acc  = max(0, 100-mape)
    return rmse, mae, r2, mape, acc

# ------------------- MAIN -------------------
if __name__ == "__main__":
    # ---- load & engineer ----
    df = load_hvac_data("hvac_dataset.csv")
    target = "Chiller Energy Consumption (kWh)"
    df, feats = add_temporal_features(df, target)

    # ---- split ----
    n = len(df)
    train_df, val_df, test_df = df[:int(0.7*n)], df[int(0.7*n):int(0.9*n)], df[int(0.9*n):]

    # ---- scale & seq ----
    look_back = 72
    (train_f, val_f, test_f,
     train_t, val_t, test_t,
     f_scaler, t_scaler) = scale_splits(train_df, val_df, test_df, feats, target)

    X_tr, y_tr = create_sequences(train_f, train_t, look_back)
    X_va, y_va = create_sequences(val_f,   val_t,   look_back)
    X_te, y_te = create_sequences(test_f,  test_t,  look_back)

    batch = 32
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                              batch_size=batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va)),
                              batch_size=batch, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)),
                              batch_size=batch, shuffle=False)

    # ---- models ----
    input_dim = X_tr.shape[2]
    models = {
        "LSTM":    LSTMNet(input_dim),
        "BiLSTM":  BiLSTMNet(input_dim),
        "GRU":     GRUNet(input_dim),
        "SSM":     SSMNet(input_dim, d_model=64, d_state=16, n_layers=3)
    }

    results, pred_dict = [], {}
    for name, net in models.items():
        print(f"\n=== Training {name} (CPU) ===")
        trained, tr_l, va_l = train_model(net, train_loader, val_loader,
                                          epochs=120,
                                          lr=5e-4 if name=="LSTM" else 1e-3)

        y_true, y_pred = evaluate_model(trained, test_loader, t_scaler)
        pred_dict[name] = y_pred
        rmse, mae, r2, mape, acc = metrics(y_true, y_pred)
        results.append([name, rmse, mae, r2, mape, acc])

        # loss curves
        plt.figure(figsize=(8,3))
        plt.plot(tr_l, label='train'); plt.plot(va_l, label='val')
        plt.title(f"{name} loss"); plt.legend(); plt.show()

        plt.figure(figsize=(12,4))
        plt.plot(y_true[:400], label='True')
        plt.plot(y_pred[:400], label='Pred')
        plt.title(f"{name} – sample"); plt.legend(); plt.show()

    # ---- ensemble (4 models) ----
    ens = np.mean(np.stack([pred_dict[k] for k in pred_dict]), axis=0)
    rmse_e, mae_e, r2_e, mape_e, acc_e = metrics(y_true, ens)
    results.append(["Ensemble", rmse_e, mae_e, r2_e, mape_e, acc_e])

    # ---- tree baselines ----
    def flatten(X): return np.hstack([X.mean(axis=1), X.std(axis=1), X[:, -1, :]])
    X_tr_f = flatten(X_tr); X_te_f = flatten(X_te)
    y_tr_f = t_scaler.inverse_transform(y_tr).ravel()
    y_te_f = t_scaler.inverse_transform(y_te).ravel()

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr_f, y_tr_f); rf_pred = rf.predict(X_te_f)
    results.append(["RandomForest"] + list(metrics(y_te_f, rf_pred)))

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1)
    xgb.fit(X_tr_f, y_tr_f); xgb_pred = xgb.predict(X_te_f)
    results.append(["XGBoost"] + list(metrics(y_te_f, xgb_pred)))

    # ---- final table & bar plot ----
    df_res = pd.DataFrame(results,
                columns=["Model","RMSE","MAE","R²","MAPE","Accuracy (%)"]).round(4)
    print("\n========== FINAL COMPARISON (CPU) ==========")
    print(df_res)

    plt.figure(figsize=(9,5))
    bars = plt.bar(df_res["Model"], df_res["Accuracy (%)"],
                   color=["#8da0cb" if "SSM" not in m else "#66c2a5" for m in df_res["Model"]],
                   edgecolor="black")
    plt.title("Forecast Accuracy (100-MAPE) – CPU"); plt.ylabel("Accuracy (%)")
    plt.ylim(0,100); plt.xticks(rotation=15); plt.tight_layout(); plt.show()
