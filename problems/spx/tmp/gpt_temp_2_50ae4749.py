import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize result Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Trend components
    sma5_close = df['close'].rolling(window=5, min_periods=5).mean()
    short_term_trend = (df['close'] - sma5_close) / sma5_close
    
    intraday_trend = (df['high'] - df['low']) / df['open']
    
    # Volume Trend components
    sma5_volume = df['volume'].rolling(window=5, min_periods=5).mean()
    volume_momentum = (df['volume'] - sma5_volume) / sma5_volume
    
    # Volume Stability calculation
    def vol_stability(x):
        return x.std() / x.mean() if x.mean() != 0 else 0
    
    volume_stability = df['volume'].rolling(window=5, min_periods=5).apply(vol_stability, raw=False)
    
    # Combine components with equal weights
    factor = (short_term_trend + intraday_trend + volume_momentum + volume_stability) / 4
    
    return factor
