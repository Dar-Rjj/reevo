import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Short-Term Reversal Signal
    reversal_raw = df['close'] - df['close'].shift(1)
    reversal_rank = reversal_raw.rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # Volume Confirmation
    volume_mean = df['volume'].shift(5).rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_mean
    close_delta = df['close'].diff()
    volume_delta = df['volume'].diff()
    volume_corr = close_delta.rolling(window=20).corr(volume_delta)

    # Combine signals
    factor = reversal_rank * volume_ratio * (1 + volume_corr)
    
    return factor
