import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Mean Reversion Signal
    sma_10 = data['close'].rolling(window=10, min_periods=10).mean()
    price_deviation = data['close'] / sma_10
    reversion_signal = -price_deviation
    
    # Trend Filter
    # Calculate rolling slope (30 days)
    trend_strength = pd.Series(index=data.index, dtype=float)
    for i in range(29, len(data)):
        window = data['close'].iloc[i-29:i+1]
        slope = linregress(np.arange(30), window.values).slope
        trend_strength.iloc[i] = slope
    
    # Avoid division by zero
    trend_strength = trend_strength.replace(0, np.nan)
    trend_adjusted = reversion_signal / trend_strength
    
    # Volume Confirmation
    avg_volume = data['volume'].rolling(window=15, min_periods=15).mean()
    std_volume = data['volume'].rolling(window=25, min_periods=25).std()
    volume_zscore = (data['volume'] - avg_volume) / std_volume
    
    # Combine components
    factor = trend_adjusted * volume_zscore
    
    return factor
