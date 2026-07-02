import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Measure Price Compression
    # Compute Intraday Range (High - Low)
    intraday_range = df['high'] - df['low']
    
    # Normalize by Volatility (Rolling 5-day StdDev of Returns)
    rolling_vol = returns.rolling(5).std()
    normalized_range = intraday_range / rolling_vol
    
    # Measure Volume Contraction
    # Compute current volume
    current_volume = df['volume']
    
    # Compare with historical volume (Rolling 10-day Volume Percentile)
    def rolling_percentile(s):
        return s.rank(pct=True).iloc[-1]
    
    volume_percentile = current_volume.rolling(10).apply(rolling_percentile, raw=False)
    
    # Combine Signals
    # Multiply Price Compression by Volume Contraction
    combined_signal = normalized_range * volume_percentile
    
    # Apply 5-day Rolling Rank
    factor = combined_signal.rolling(5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    return factor
