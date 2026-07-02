import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Short-Term Reversal Signal
    delta_close = df['close'] - df['close'].shift(3)
    reversal_raw = -delta_close  # Negative sign for reversal
    
    # Rolling rank of reversal_raw over a window of 10
    rolling_rank = reversal_raw.rolling(window=10).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    # Volume Confirmation
    volume_rolling_mean = df['volume'].shift(1).rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_rolling_mean
    
    # Correlation between absolute delta(close) and delta(volume)
    abs_delta_close = abs(df['close'] - df['close'].shift(1))
    delta_volume = df['volume'] - df['volume'].shift(1)
    correlation = abs_delta_close.rolling(window=10).corr(delta_volume)
    
    # Combine signals
    price_reversal_intensity = rolling_rank * volume_ratio * correlation
    
    return price_reversal_intensity
