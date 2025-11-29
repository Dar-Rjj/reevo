import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    # Morning Range: (High - Open) / Open
    morning_range = (data['high'] - data['open']) / data['open']
    
    # Afternoon Range: (Close - Low) / Low  
    afternoon_range = (data['close'] - data['low']) / data['low']
    
    # Compression Factor: (Close - Open) × (Morning_Range / Afternoon_Range)
    compression_factor = (data['close'] - data['open']) * (morning_range / afternoon_range.replace(0, np.nan))
    
    # Volume clustering components
    volume_quantile_80 = data['volume'].rolling(window=20, min_periods=10).quantile(0.8)
    volume_quantile_20 = data['volume'].rolling(window=20, min_periods=10).quantile(0.2)
    
    high_volume_cluster = data['volume'] > volume_quantile_80
    low_volume_cluster = data['volume'] < volume_quantile_20
    
    # Calculate volume cluster duration
    volume_cluster_duration = pd.Series(index=data.index, dtype=float)
    current_duration = 0
    for i in range(len(data)):
        if high_volume_cluster.iloc[i] or low_volume_cluster.iloc[i]:
            current_duration += 1
        else:
            current_duration = 0
        volume_cluster_duration.iloc[i] = current_duration
    
    # Reversal Factor: (Close - Open) × Volume_Cluster_Duration
    reversal_factor = (data['close'] - data['open']) * volume_cluster_duration
    
    # Opening Momentum Persistence
    prev_close = data['close'].shift(1)
    prev_high_low_range = data['high'].shift(1) - data['low'].shift(1)
    early_momentum = (data['open'] - prev_close) / prev_high_low_range.replace(0, np.nan)
    
    # Persistence Factor: (Close - Open) × Early_Momentum
    persistence_factor = (data['close'] - data['open']) * early_momentum
    
    # Price-Range Efficiency
    high_low_range = data['high'] - data['low']
    efficiency_factor = (data['close'] - data['open']) / high_low_range.replace(0, np.nan)
    
    # Intraday Support Resistance
    support_level = data['low'].rolling(window=5, min_periods=3).min()
    resistance_level = data['high'].rolling(window=5, min_periods=3).max()
    
    # Breakout Factor: (Close - Support) / (Resistance - Support)
    resistance_support_diff = resistance_level - support_level
    breakout_factor = (data['close'] - support_level) / resistance_support_diff.replace(0, np.nan)
    
    # Volume Asymmetry Momentum
    # Estimate morning volume (first half) and afternoon volume (second half)
    # Using amount as proxy for intraday volume distribution
    total_volume = data['volume']
    asymmetry = (data['amount'].rolling(window=5, min_periods=3).mean() - 
                data['amount'].shift(1).rolling(window=5, min_periods=3).mean()) / total_volume.replace(0, np.nan)
    
    mid_price = (data['high'] + data['low']) / 2
    asymmetry_factor = asymmetry * (data['close'] - mid_price)
    
    # False Breakout Detection
    breakout_test = (data['high'] > resistance_level) | (data['low'] < support_level)
    false_breakout_factor = (data['close'] - data['open']) / high_low_range.replace(0, np.nan)
    false_breakout_factor = false_breakout_factor.where(breakout_test, 0)
    
    # Price-Volume Synchronization
    price_movement = (data['close'] - data['open']) / data['open'].replace(0, np.nan)
    avg_volume = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_intensity = data['volume'] / avg_volume.replace(0, np.nan)
    synchronization_factor = price_movement * volume_intensity
    
    # Combine all factors with equal weights
    factors = [
        compression_factor,
        reversal_factor,
        persistence_factor,
        efficiency_factor,
        breakout_factor,
        asymmetry_factor,
        false_breakout_factor,
        synchronization_factor
    ]
    
    # Standardize each factor and combine
    for i in range(len(data)):
        if i >= 20:  # Ensure enough data for meaningful calculations
            day_factors = []
            for factor in factors:
                if not pd.isna(factor.iloc[i]):
                    # Use rolling z-score for standardization
                    rolling_mean = factor.iloc[max(0, i-19):i+1].mean()
                    rolling_std = factor.iloc[max(0, i-19):i+1].std()
                    if rolling_std > 0:
                        standardized = (factor.iloc[i] - rolling_mean) / rolling_std
                        day_factors.append(standardized)
            
            if day_factors:
                result.iloc[i] = np.mean(day_factors)
    
    return result
