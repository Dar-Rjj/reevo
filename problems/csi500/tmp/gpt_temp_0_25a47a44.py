import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Efficiency Signal
    # Daily Efficiency: (close - open) / (high - low)
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Efficiency Momentum: slope of Daily Efficiency over 5 days
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    efficiency_momentum = daily_efficiency.rolling(window=5, min_periods=5).apply(calc_slope, raw=True)
    
    # Volume Confirmation
    # Volume Ratio: volume on up days / volume on down days
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    # Calculate rolling volume ratio over 10 days
    def volume_ratio_func(window):
        if len(window) < 10:
            return np.nan
        up_volume = window[up_days[window.index]].sum()
        down_volume = window[down_days[window.index]].sum()
        if down_volume == 0:
            return np.nan
        return up_volume / down_volume
    
    volume_ratio = data['volume'].rolling(window=10, min_periods=10).apply(volume_ratio_func, raw=False)
    
    # Volume Stability: std(volume) / mean(volume) over 10 days
    volume_stability = data['volume'].rolling(window=10, min_periods=10).std() / data['volume'].rolling(window=10, min_periods=10).mean()
    
    # Combined Alpha
    # Efficiency-Volume Composite: Daily Efficiency × Volume Ratio
    efficiency_volume_composite = daily_efficiency * volume_ratio
    
    # Momentum-Stability Signal: Efficiency Momentum × Volume Stability
    momentum_stability_signal = efficiency_momentum * volume_stability
    
    # Final alpha factor: combine both signals
    alpha_factor = efficiency_volume_composite + momentum_stability_signal
    
    return alpha_factor
