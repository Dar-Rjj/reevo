import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate Intraday Return
    intraday_return = (data['close'] - data['open']) / data['open']
    
    # Normalize by Price Range
    price_range = data['high'] - data['low']
    price_efficiency = intraday_return / price_range
    
    # Calculate Volume Percentile
    rolling_volume = data['volume'].rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Apply Min-Max Scaling
    min_volume = rolling_volume.rolling(window=20, min_periods=1).min()
    max_volume = rolling_volume.rolling(window=20, min_periods=1).max()
    volume_score = 2 * (rolling_volume - min_volume) / (max_volume - min_volume)
    
    # Combine Signals
    combined_signal = price_efficiency * volume_score
    
    # Apply Directional Adjustment
    directional_adjustment = combined_signal * intraday_return.apply(lambda x: 1 if x > 0 else -1)
    
    return directional_adjustment
