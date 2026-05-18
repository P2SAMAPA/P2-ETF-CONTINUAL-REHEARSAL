import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn
import config
import data_manager
from continual_model import ContinualLearner

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

def create_features_and_target(returns_df, macro_df, etf, lookback=5):
    ret = returns_df[etf]
    data = pd.DataFrame(index=ret.index)
    for lag in range(1, lookback+1):
        data[f'lag_{lag}'] = ret.shift(lag)
    for col in macro_df.columns:
        data[col] = macro_df[col]
    data['target'] = ret.shift(-1)
    data = data.dropna()
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return X, y

def train_continual_model(returns_df, macro_df, etf, window, method='replay', buffer_size=200, ewc_lambda=0.1):
    if len(returns_df) < window + 20:
        return 0.0
    ret_win = returns_df.iloc[-window:]
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret_win.index, columns=config.MACRO_COLUMNS)
    common = ret_win.index.intersection(macro_win.index)
    ret_win = ret_win.loc[common]
    macro_win = macro_win.loc[common]
    lookback = 5
    X_list, y_list = [], []
    for i in range(lookback, len(ret_win)-1):
        X_row = [ret_win[etf].iloc[i-lag] for lag in range(1, lookback+1)]
        X_row.extend(macro_win.iloc[i].values)
        X_list.append(X_row)
        y_list.append(ret_win[etf].iloc[i+1])
    if len(X_list) < 10:
        return 0.0
    X = np.array(X_list)
    y = np.array(y_list)
    input_dim = X.shape[1]
    learner = ContinualLearner(input_dim, method=method, replay_buffer_size=buffer_size, ewc_lambda=ewc_lambda)
    # Train on all data (not truly continual, but for simplicity)
    X_t = torch.tensor(X, dtype=torch.float32).to(learner.device)
    y_t = torch.tensor(y, dtype=torch.float32).to(learner.device)
    learner.model.train()
    learner.optimizer.zero_grad()
    outputs = learner.model(X_t).squeeze()
    loss = nn.MSELoss()(outputs, y_t)
    loss.backward()
    learner.optimizer.step()
    last_X = X[-1:].reshape(1, -1)
    pred = learner.predict(last_X)
    return pred

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Continual Rehearsal) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 20:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 20:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_scores = {}
            for etf in tickers:
                pred = train_continual_model(returns, macro, etf, win,
                                             method=config.METHOD,
                                             buffer_size=config.REPLAY_BUFFER_SIZE,
                                             ewc_lambda=config.EWC_LAMBDA)
                etf_scores[etf] = pred
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by continual learning prediction: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/continual_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Continual Rehearsal Engine complete ===")

if __name__ == "__main__":
    main()
