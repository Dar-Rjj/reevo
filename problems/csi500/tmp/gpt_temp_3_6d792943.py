import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate features
    data['range'] = data['high'] - data['low']
    data['close_position'] = (data['close'] - data['low']) / np.where(data['range'] == 0, 1, data['range'])
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Calculate rolling statistics
    data['range_3d_avg'] = data['range'].rolling(window=3, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1).replace(0, 1)
    data['true_range'] = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
    data['volatility_3d_avg'] = data['true_range'].rolling(window=3, min_periods=1).mean()
    
    for i in range(len(data)):
        if i < 3:  # Skip first few days for sufficient data
            factor.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev1 = data.iloc[i-1]
        prev2 = data.iloc[i-2]
        
        # Factor 1: Intraday Momentum-Range Divergence
        # Range efficiency
        range_efficiency = current['close_position'] if current['range'] > 0 else 0.5
        range_efficiency_3d = (data.iloc[i-2:i+1]['close_position'].mean() if 
                              len(data.iloc[i-2:i+1]) == 3 else 0.5)
        range_efficiency_div = range_efficiency - range_efficiency_3d
        
        # Momentum shifts
        morning_momentum = ((current['high'] - current['open']) / 
                           np.where(current['open'] - current['low'] == 0, 1, 
                                   current['open'] - current['low']))
        afternoon_momentum = ((current['close'] - current['mid_price']) / 
                             np.where(current['mid_price'] - current['low'] == 0, 1, 
                                     current['mid_price'] - current['low']))
        momentum_shift = afternoon_momentum - morning_momentum
        
        # Volume confirmation
        volume_confirmation = np.log(current['volume_ratio']) if current['volume_ratio'] > 0 else 0
        
        # Divergence strength
        divergence_strength = range_efficiency_div * momentum_shift * volume_confirmation
        
        factor1 = divergence_strength * np.sign(divergence_strength) * abs(volume_confirmation)
        
        # Factor 2: Gap-Pressure Range Alignment
        gap_magnitude = (current['open'] / prev1['close'] - 1) if prev1['close'] > 0 else 0
        buying_pressure = current['close_position']
        net_pressure = gap_magnitude * buying_pressure
        
        # Range support
        range_confirmation = (current['range'] / current['range_3d_avg'] if 
                             current['range_3d_avg'] > 0 else 1)
        range_strength = range_confirmation * np.sign(gap_magnitude)
        
        # Pressure-range alignment
        alignment = net_pressure * range_strength
        
        factor2 = alignment * abs(range_strength) * np.sign(alignment)
        
        # Factor 3: Volume-Volatility Breakout Sensitivity
        volatility_compression = (current['true_range'] / current['volatility_3d_avg'] if 
                                 current['volatility_3d_avg'] > 0 else 1)
        compression_ratio = 1 / volatility_compression if volatility_compression > 0 else 1
        
        # Volume sensitivity
        price_movement = abs(current['close'] - current['open'])
        volume_sensitivity = (current['volume'] / price_movement if price_movement > 0 else 
                             current['volume'])
        
        # Breakout signal
        breakout_signal = compression_ratio * volume_sensitivity
        
        factor3 = breakout_signal * np.sign(breakout_signal) * abs(compression_ratio - 1)
        
        # Factor 4: Cluster-Gap Volatility Momentum
        # Volume clustering
        volume_changes = [data.iloc[j]['volume_ratio'] for j in range(i-2, i+1)]
        cluster_strength = np.std(volume_changes) if len(volume_changes) == 3 else 0
        
        # Gap alignment
        gap_direction = np.sign(current['open'] - prev1['close'])
        gap_magnitude_abs = abs(current['open'] - prev1['close']) / prev1['close'] if prev1['close'] > 0 else 0
        
        # Cluster-gap momentum
        cluster_gap_momentum = cluster_strength * gap_direction * gap_magnitude_abs
        
        factor4 = cluster_gap_momentum * abs(cluster_strength) * np.sign(cluster_gap_momentum)
        
        # Combine factors with equal weighting
        combined_factor = (factor1 + factor2 + factor3 + factor4) / 4
        
        factor.iloc[i] = combined_factor
    
    # Fill NaN values
    factor = factor.fillna(0)
    
    return factor
