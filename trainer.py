import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import config
import data_manager
from continual_learner import MLP, ReplayBuffer, compute_fisher_information, train_step_replay, train_step_ewc

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_features(df, lag=1):
    """Use lagged returns as features, plus macro levels."""
    # df has both ETF returns and macro columns
    # We'll create features for the most recent day only (to predict next day)
    # For training, we create a sequence: for each day, features = returns of past `lag` days + macro levels
    # But we simplify: features = last day's returns + macro levels
    # Better: use window of past returns. For speed, we'll use only the last day's returns as features.
    etf_cols = [c for c in df.columns if c not in config.MACRO_COLUMNS]
    macro_cols = config.MACRO_COLUMNS
    X = []
    y = []
    for i in range(1, len(df)):
        # Features: previous day's returns of all ETFs + macro levels of that day
        prev_returns = df[etf_cols].iloc[i-1].values
        prev_macro = df[macro_cols].iloc[i-1].values if macro_cols else np.array([])
        X.append(np.concatenate([prev_returns, prev_macro]))
        # Target: next day's returns of all ETFs (we predict each ETF separately? Actually we need a multi-output model)
        # For simplicity, we'll train one model per ETF (like before)
        y_target = df[etf_cols].iloc[i].values
        # We'll loop over ETFs later.
    return np.array(X), np.array(y_target)  # but this returns y as array of all ETFs – not correct for per-ETF model.
    # We'll restructure: train a model for each ETF individually, using features that include returns of all ETFs (cross-sectional).
    # That's more complex. To keep it simple, we'll train per-ETF with features = past return of that ETF + macro.
    # That's less powerful but works for demonstration.

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Continual Rehearsal) ===")
        combined = data_manager.prepare_combined_data(df, tickers)
        if combined.empty or len(combined) < 100:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # For each ETF, build features (lagged returns + macro) and train/predict incrementally
        etf_cols = [c for c in combined.columns if c not in config.MACRO_COLUMNS]
        macro_cols = config.MACRO_COLUMNS
        if not macro_cols:
            macro_cols = []

        best_per_etf = {}
        # We'll simulate continual learning: process days one by one, updating model.
        # For the final prediction, we use the model after all updates.
        for etf in tickers:
            if etf not in etf_cols:
                continue
            # Prepare sequential data
            X_seq = []
            y_seq = []
            for i in range(1, len(combined)):
                # Features: previous day's return of this ETF + macro levels
                prev_return = combined[etf].iloc[i-1]
                prev_macro = combined[macro_cols].iloc[i-1].values if macro_cols else np.array([])
                X_seq.append(np.concatenate([[prev_return], prev_macro]))
                y_seq.append(combined[etf].iloc[i])  # next day's return
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            if len(X_seq) < 20:
                continue
            # Standardise
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_seq)
            # Convert to torch
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            y_t = torch.tensor(y_seq, dtype=torch.float32)
            # Model
            input_dim = X_t.shape[1]
            model = MLP(input_dim, config.HIDDEN_DIM, 1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.MSELoss()
            # Choose continual learning method
            if config.METHOD == "replay":
                replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
                # Train sequentially
                for i in range(len(X_t)):
                    X_single = X_t[i:i+1]
                    y_single = y_t[i:i+1]
                    train_step_replay(model, optimizer, criterion, X_single, y_single, replay_buffer, config.BATCH_SIZE, device)
            else:  # ewc
                # First, train on initial batch (first 20% of data) to compute Fisher
                split = int(0.2 * len(X_t))
                X_initial = X_t[:split]
                y_initial = y_t[:split]
                for i in range(0, split, config.BATCH_SIZE):
                    Xb = X_initial[i:i+config.BATCH_SIZE]
                    yb = y_initial[i:i+config.BATCH_SIZE]
                    optimizer.zero_grad()
                    pred = model(Xb).squeeze()
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                # Compute Fisher information on initial data
                # Create a DataLoader
                from torch.utils.data import TensorDataset, DataLoader
                init_dataset = TensorDataset(X_initial, y_initial)
                init_loader = DataLoader(init_dataset, batch_size=32, shuffle=True)
                fisher = compute_fisher_information(model, init_loader, device)
                old_params = {name: param.detach().clone() for name, param in model.named_parameters()}
                # Train on remaining data with EWC penalty
                for i in range(split, len(X_t)):
                    X_single = X_t[i:i+1]
                    y_single = y_t[i:i+1]
                    train_step_ewc(model, optimizer, criterion, X_single, y_single, config.EWC_LAMBDA, fisher, old_params, device)
            # After training, predict the next return (for the last available feature vector)
            # The most recent feature vector is the last row of X_scaled
            last_X = X_scaled[-1:].reshape(1, -1)
            last_X_t = torch.tensor(last_X, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                pred = model(last_X_t).item()
            best_per_etf[etf] = pred

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in combined.columns:
                    mean_ret = combined[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = max(mean_ret, 1e-6)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score)} for ticker, score in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score)} for ticker, score in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by continual prediction: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/continual_rehearsal_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Continual Rehearsal Engine complete ===")

if __name__ == "__main__":
    main()
