import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Efficiency Signal
    # Daily Efficiency Ratio: (close - open) / (high - low)
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # 5-day Efficiency Momentum: slope of Daily Efficiency Ratio
    def calc_efficiency_momentum(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        mask = ~np.isnan(series)
        if np.sum(mask) < 2:
            return np.nan
        slope, _, _, _, _ = linregress(x[mask], series[mask])
        return slope
    
    efficiency_momentum = daily_efficiency.rolling(window=5, min_periods=5).apply(
        calc_efficiency_momentum, raw=False
    )
    
    # Volume Confirmation Pattern
    # High-Low Volume Ratio: volume on up days / volume on down days
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    # Calculate rolling mean volume for up and down days separately
    up_volume = data['volume'].where(up_days).rolling(window=10, min_periods=5).mean()
    down_volume = data['volume'].where(down_days).rolling(window=10, min_periods=5).mean()
    
    high_low_volume_ratio = up_volume / down_volume
    high_low_volume_ratio = high_low_volume_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Volume Concentration: std(volume) / mean(volume) over 10 days
    volume_mean = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_std = data['volume'].rolling(window=10, min_periods=5).std()
    volume_concentration = volume_std / volume_mean
    volume_concentration = volume_concentration.replace([np.inf, -np.inf], np.nan)
    
    # Combined Alpha Factor
    # Efficiency-Volume Composite: Daily Efficiency Ratio × High-Low Volume Ratio
    efficiency_volume_composite = daily_efficiency * high_low_volume_ratio
    
    # Momentum-Confirmation Signal: 5-day Efficiency Momentum × Volume Concentration
    momentum_confirmation = efficiency_momentum * volume_concentration
    
    # Final alpha factor: average of both components
    alpha_factor = (efficiency_volume_composite + momentum_confirmation) / 2
    
    return alpha_factor
