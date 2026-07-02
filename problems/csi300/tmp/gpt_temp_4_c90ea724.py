import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Price Efficiency Component
    # Calculate 10-day moving average of Close
    ma_10 = data['close'].rolling(window=10, min_periods=1).mean()
    
    # Calculate Price Deviation
    price_deviation = (data['close'] - ma_10) / data['close']
    
    # Calculate True Range
    prev_close = data['close'].shift(1)
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hcp': abs(data['high'] - prev_close),
        'lcp': abs(data['low'] - prev_close)
    }).max(axis=1)
    
    # Calculate 10-day Average True Range
    atr_10 = tr.rolling(window=10, min_periods=1).mean()
    
    # Normalize Price Deviation by ATR
    normalized_efficiency = price_deviation / atr_10
    
    # Liquidity Confirmation
    # Compute Dollar Volume
    dollar_volume = data['volume'] * data['close']
    
    # Calculate 20-day percentile rank of Dollar Volume
    liquidity_score = dollar_volume.rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Combine Signals
    factor = normalized_efficiency * liquidity_score
    
    return factor
