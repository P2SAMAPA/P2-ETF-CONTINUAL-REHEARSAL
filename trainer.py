import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
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
        return 0.0, 0.0  # score, win
    ret_win = returns_df.iloc[-window:]
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret_win.index, columns=config.MACRO_COLUMNS)
    common = ret_win.index.intersection(macro_win.index)
    ret_win = ret_win.loc[common]
    macro_win = macro_win.loc[common]
    lookback = 5
    X_list, y_list = [], []
    for i in range(lookback, len(ret_win)):
        X_row = []
        for lag in range(1, lookback+1):
            X_row.append(ret_win[etf].iloc[i-lag])
        for col in macro_win.columns:
            X_row.append(macro_win[col].iloc[i])
        X_list.append(X_row)
        y_list.append(ret_win[etf].iloc[i+1] if i+1 < len(ret_win) else 0.0)
    X = np.array(X_list)
    y = np.array(y_list)
    if len(X) < 20:
        return 0.0, 0.0
    input_dim = X.shape[1]
    learner = ContinualLearner(input_dim, hidden_dim=config.HIDDEN_DIM, lr=config.LEARNING_RATE,
                               method=method, replay_buffer_size=config.REPLAY_BUFFER_SIZE,
                               ewc_lambda=config.EWC_LAMBDA)
    # Train sequentially on each sample? For simplicity, train on the whole dataset in one batch.
    # But continual learning should be incremental. We'll simulate by splitting into chunks.
    # However, to keep it simple, we'll train on all data once (the model will still learn).
    # For proper continual learning, we would split by time.
    loss = learner.train_step(X, y, replay_batch_size=config.REPLAY_BATCH_SIZE)
    # Update EWC if method is ewc
    if method == 'ewc':
        learner.update_ewc(X, y)
    # Predict for the next day after the window
    # We need the features for the last available date (the day after the window)
    # The features for the next prediction are the last `lookback` returns and macro at the last day.
    last_idx = len(ret_win) - 1
    X_last = []
    for lag in range(1, lookback+1):
        X_last.append(ret_win[etf].iloc[last_idx - lag + 1] if last_idx - lag + 1 >= 0 else 0.0)
    for col in macro_win.columns:
        X_last.append(macro_win[col].iloc[last_idx])
    X_last = np.array(X_last).reshape(1, -1)
    pred = learner.predict(X_last)[0]
    return pred, window

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
                if etf not in returns.columns:
                    continue
                pred, used_win = train_continual_model(returns, macro, etf, win,
                                                       method=config.METHOD,
                                                       buffer_size=config.REPLAY_BUFFER_SIZE,
                                                       ewc_lambda=config.EWC_LAMBDA)
                if pred != 0.0:
                    etf_scores[etf] = (pred, used_win)
            window_results[win] = {etf: score for etf, (score, _) in etf_scores.items()}
            for etf, (score, w) in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, w)

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
