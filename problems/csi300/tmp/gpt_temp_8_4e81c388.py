import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Momentum Signal
    price_change = (df['close'] - df['open']) / df['open']
    price_range = (df['high'] - df['low']) / df['close']
    momentum_signal = price_change / price_range
    
    # Apply Volume Filter
    rolling_volume_mean = df['volume'].rolling(window=15, min_periods=1).mean()
    volume_ratio = df['volume'] / rolling_volume_mean
    
    # Compute Rolling Volume Percentile
    volume_percentile = df['volume'].rolling(window=15, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Final Factor
    factor = momentum_signal * volume_ratio * volume_percentile
    
    return factor
