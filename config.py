import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-continual-rehearsal-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# Continual learning method: "replay" or "ewc"
METHOD = "replay"

# Experience replay parameters
REPLAY_BUFFER_SIZE = 200
REPLAY_BATCH_SIZE = 32

# EWC parameters
EWC_LAMBDA = 0.1

# Model hyperparameters
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS_PER_BATCH = 1

TOP_N = 3
