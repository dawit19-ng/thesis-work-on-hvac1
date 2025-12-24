import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os, random, math, warnings
warnings.filterwarnings("ignore")

# ------------------- Reproducibility -------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ------------------- Load Data -------------------
def load_hvac_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Local Time (Timezone : GMT+8h)"])
    df.rename(columns={"Local Time (Timezone : GMT+8h)": "timestamp"}, inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ------------------- Feature Engineering -------------------
def add_temporal_features(df, target):
    features = [
        "Cooling Water Temperature (C)", "Humidity (%)", "Building Load (RT)",
        "Chilled Water Rate (L/sec)", "Outside Temperature (F)", "Dew Point (F)",
        "Wind Speed (mph)", "Pressure (in)", "hour", "day_of_week"
    ]
    for lag in [1, 2, 3]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
        features.append(f"{target}_lag{lag}")

    for window in [3, 6]:
        df[f"{target}_roll_mean_{window}"] = df[target].rolling(window).mean()
        df[f"{target}_roll_std_{window}"] = df[target].rolling(window).std()
        features.extend([f"{target}_roll_mean_{window}", f"{target}_roll_std_{window}"])
#missing value via forward and backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    #target= chiller energy consumption(kw)
    df[target] = df[target].clip(lower=df[target].quantile(0.01),
                                 upper=df[target].quantile(0.99))
    return df, features

# ------------------- Sequence Creation -------------------
def create_sequences(features_scaled, target_scaled, look_back=72):
    X, y = [], []
    for i in range(look_back, len(features_scaled)):
        X.append(features_scaled[i - look_back:i])
        y.append(target_scaled[i])
    return np.array(X), np.array(y)

# ------------------- Scaling -------------------
def scale_splits(train_df, val_df, test_df, features, target):
    feature_scaler = MinMaxScaler().fit(train_df[features])
    target_scaler = MinMaxScaler().fit(train_df[[target]])

    train_f = feature_scaler.transform(train_df[features])
    val_f   = feature_scaler.transform(val_df[features])
    test_f  = feature_scaler.transform(test_df[features])

    train_t = target_scaler.transform(train_df[[target]])
    val_t   = target_scaler.transform(val_df[[target]])
    test_t  = target_scaler.transform(test_df[[target]])

    return train_f, val_f, test_f, train_t, val_t, test_t, feature_scaler, target_scaler

# ------------------- PyTorch Model Builders -------------------
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True,
                             dropout=0, num_layers=1)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True,
                             dropout=0.0, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class BiLSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_dim, hidden, batch_first=True,
                               bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden*2)
        self.dropout1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(hidden*2, hidden//2, batch_first=True,
                               bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out = self.bn(out[:, -1, :])
        out = self.dropout1(out)
        out, _ = self.bilstm2(out.unsqueeze(1))
        out = self.dropout2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden, batch_first=True,
                           dropout=0.2, num_layers=1)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True,
                           dropout=0.0, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ------------------- SSM/Mamba-inspired Model -------------------
class SSMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            h = torch.tanh(h @ self.A + x[:, t, :] @ self.B)
            out = h @ self.C
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class SSMNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.ssm  = SSMLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ssm(out)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ------------------- Training Loop -------------------
def train_model(model, train_loader, val_loader,
                epochs=100, lr=0.001, patience_es=10, patience_lr=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience_lr,
        min_lr=1e-6,
        threshold=1e-4
    )
    best_val = float('inf')
    wait_es = 0

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait_es = 0
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        else:
            wait_es += 1
            if wait_es >= patience_es:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train loss: {epoch_loss:.6f} | Val loss: {val_loss:.6f}")

    model.load_state_dict(torch.load(f"{model.__class__.__name__}_best.pth"))
    return model, train_losses, val_losses

# ------------------- Evaluation -------------------
@torch.no_grad()
def evaluate_model(model, test_loader, target_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    trues = []
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        preds.append(out)
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_true_inv = target_scaler.inverse_transform(y_true)

    rmse = math.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    r2   = r2_score(y_true_inv, y_pred_inv)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) /
                          np.where(y_true_inv == 0, 1e-10, y_true_inv))) * 100
    acc  = max(0, 100 - mape)
    return y_true_inv, y_pred_inv, rmse, mae, r2, mape, acc

# ------------------- Baseline Evaluation -------------------
def evaluate_baseline(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    acc  = max(0, 100 - mape)
    return rmse, mae, r2, mape, acc

# ================================================================
#                           MAIN SCRIPT
# ================================================================
# ================================================================
# MAIN SCRIPT — FULL SEED EVALUATION (10 seeds)
# ================================================================
if __name__ == "__main__":
    filepath = "hvac_dataset.csv"
    df = load_hvac_data(filepath)
    target = "Chiller Energy Consumption (kWh)"
    df, features = add_temporal_features(df, target)

    # Fixed splits (identical for all seeds)
    split_1 = int(0.7 * len(df))
    split_2 = int(0.9 * len(df))
    train_df, val_df, test_df = df[:split_1], df[split_1:split_2], df[split_2:]

    look_back = 72  # ← FIXED BEST VALUE
    batch_size = 16
    SEEDS = [42, 123, 456, 789, 1010, 2021, 2022, 1337, 777, 999]

    all_results = []
    print("Starting 10-seed evaluation (look-back = 72h)...\n")

    for seed in SEEDS:
        print(f"\n{'=' * 20} SEED {seed} {'=' * 20}")
        set_seed(seed)

        # Scaling & sequences (fit scalers only on train)
        train_f, val_f, test_f, train_t, val_t, test_t, f_scaler, t_scaler = scale_splits(
            train_df, val_df, test_df, features, target)

        X_train, y_train = create_sequences(train_f, train_t, look_back)
        X_val, y_val = create_sequences(val_f, val_t, look_back)
        X_test, y_test = create_sequences(test_f, test_t, look_back)

        # DataLoaders
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                                batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
                                 batch_size=batch_size, shuffle=False)

        input_dim = X_train.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store predictions for ensemble
        deep_preds = {}
        y_true_inv = None

        # Train each deep model
        for name, model_cls in [
            ("LSTM", lambda: LSTMNet(input_dim)),
            ("BiLSTM", lambda: BiLSTMNet(input_dim)),
            ("GRU", lambda: GRUNet(input_dim)),
            ("SSM", lambda: SSMNet(input_dim))
        ]:
            set_seed(seed)
            model = model_cls().to(device)
            print(f"  → Training {name}...")
            trained, _, _ = train_model(model, train_loader, val_loader,
                                        epochs=120, lr=0.001, patience_es=15)

            y_true_np, y_pred_np, rmse, mae, r2, mape, acc = evaluate_model(trained, test_loader, t_scaler)
            if y_true_inv is None:
                y_true_inv = y_true_np.flatten()
            deep_preds[name] = y_pred_np.flatten()

            all_results.append({"seed": seed, "model": name,
                                "RMSE": rmse, "MAE": mae, "R2": r2, "Accuracy": acc})

        # Ensemble of 4 deep models
        ens_pred = np.mean(np.stack(list(deep_preds.values())), axis=0)
        ens_rmse, ens_mae, ens_r2, ens_mape, ens_acc = evaluate_baseline(y_true_inv, ens_pred)
        all_results.append({"seed": seed, "model": "Ensemble",
                            "RMSE": ens_rmse, "MAE": ens_mae, "R2": ens_r2, "Accuracy": ens_acc})


        # Tree baselines (same seed & same flattened features)
        def flatten(X):
            return np.hstack([X.mean(axis=1), X.std(axis=1), X[:, -1, :]])


        X_train_flat = flatten(X_train)
        X_test_flat = flatten(X_test)
        y_train_f = y_train.ravel()
        y_test_f = y_test.ravel()

        # Random Forest
        rf = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
        rf.fit(X_train_flat, y_train_f)
        rf_pred = rf.predict(X_test_flat)
        rf_metrics = evaluate_baseline(y_true_inv, rf_pred)
        all_results.append(
            {"seed": seed, "model": "RandomForest", **dict(zip(["RMSE", "MAE", "R2", "MAPE", "Accuracy"], rf_metrics))})

        # XGBoost
        xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        xgb.fit(X_train_flat, y_train_f)
        xgb_pred = xgb.predict(X_test_flat)
        xgb_metrics = evaluate_baseline(y_true_inv, xgb_pred)
        all_results.append(
            {"seed": seed, "model": "XGBoost", **dict(zip(["RMSE", "MAE", "R2", "MAPE", "Accuracy"], xgb_metrics))})

    # ==================== FINAL RESULTS ====================
    df_results = pd.DataFrame(all_results)
    summary = df_results.groupby("model")[["RMSE", "MAE", "R2", "Accuracy"]].agg(["mean", "std"]).round(4)
    print("\n" + "=" * 70)
    print("FINAL 10-SEED RESULTS (look-back = 72h)")
    print("=" * 70)
    print(summary)

    # Paired t-test: Ensemble vs XGBoost (on RMSE)
    ens_rmse = df_results[df_results.model == "Ensemble"]["RMSE"]
    xgb_rmse = df_results[df_results.model == "XGBoost"]["RMSE"]
    t_stat, p_val = stats.ttest_rel(ens_rmse, xgb_rmse)
    print(f"\nPaired t-test (Ensemble vs XGBoost) → t = {t_stat:.3f}, p = {p_val:.2e}")

    # Save everything
    summary.to_csv("MAIN_RESULTS_10seeds_72h.csv")
    df_results.to_csv("all_individual_runs.csv", index=False)

    # Beautiful plot for thesis
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x="model", y="Accuracy", errorbar="sd", capsize=0.1, palette="tab10")
    plt.title("HVAC Chiller Energy Prediction — 10 Random Seeds (72h context)")
    plt.ylabel("Accuracy = 100 − MAPE (%)")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("Main_Results_10seeds.pdf", dpi=300, bbox_inches="tight")
    plt.show()