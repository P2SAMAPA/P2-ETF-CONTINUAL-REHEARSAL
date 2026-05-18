import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_combined_data(df, universe_tickers):
    # ETF returns
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    # Macro data
    macro = df[config.MACRO_COLUMNS].copy() if all(c in df.columns for c in config.MACRO_COLUMNS) else pd.DataFrame()
    macro = macro.ffill().fillna(0)
    combined = pd.concat([returns, macro], axis=1).dropna()
    return combined
