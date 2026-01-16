import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    for t in range(len(df)):
        if t < 10:  # Need at least 10 periods for calculations
            factor.iloc[t] = np.nan
            continue
            
        current = df.iloc[t]
        past = df.iloc[:t+1]  # Only use current and past data
        
        # 1. Price Midpoint Divergence
        midpoint = (current['high'] + current['low']) / 2
        price_divergence = (midpoint - current['close']) / current['close']
        
        # 2. Volume Surge Amplification
        volume_window = past['volume'].iloc[t-9:t+1]
        volume_zscore = (current['volume'] - volume_window.mean()) / (volume_window.std() + 1e-6)
        amplified_signal = price_divergence * volume_zscore
        
        # 3. Normalize by Price Volatility
        close_window = past['close'].iloc[t-4:t+1]
        price_volatility = close_window.std()
        normalized_signal = amplified_signal / (price_volatility + 1e-6)
        
        # 4. Trend-Adjusted Momentum
        if t >= 10:  # Need at least 10 periods for SMA calculations
            sma5 = past['close'].iloc[t-4:t+1].mean()
            sma10 = past['close'].iloc[t-9:t+1].mean()
            trend = sma5 - sma10
            final_factor = normalized_signal * trend
        else:
            final_factor = np.nan
            
        factor.iloc[t] = final_factor
    
    return factor
