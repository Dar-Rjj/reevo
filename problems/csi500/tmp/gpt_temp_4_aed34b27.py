import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Compute Momentum-Reversal Divergence Score
    # Raw Momentum Divergence Component
    high_low_range = data['high'] - data['low'] + 1e-12
    upward_momentum_efficiency = (data['high'] - data['open']) / high_low_range
    downward_momentum_efficiency = (data['open'] - data['low']) / high_low_range
    momentum_divergence = (upward_momentum_efficiency - downward_momentum_efficiency) * (data['close'] - data['open']) / high_low_range
    
    # Reversal Persistence Component
    intraday_reversal_pressure = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-12)
    reversal_efficiency = (data['close'] - data['open']) / high_low_range
    reversal_persistence = intraday_reversal_pressure * reversal_efficiency
    
    # Combine Momentum and Reversal Components
    momentum_reversal_divergence = momentum_divergence * reversal_persistence
    # Apply 5-day exponential smoothing
    momentum_reversal_signal = momentum_reversal_divergence.ewm(span=5, adjust=False).mean()
    
    # Assess Volume Acceleration-Persistence Dynamics
    # Volume Acceleration Component
    volume_change_rate = data['volume'] / data['volume'].shift(1)
    volume_trend_persistence = volume_change_rate / data['volume'].rolling(5).std()
    volume_acceleration = volume_change_rate * volume_trend_persistence
    
    # Volume Persistence Confirmation
    volume_stability_ratio = data['volume'] / data['volume'].rolling(3).mean()
    range_efficiency = (data['high'] - data['low']) / (data['open'] + 1e-12)
    volume_persistence = volume_stability_ratio * range_efficiency
    
    # Combine Volume Components with directional alignment
    volume_dynamics = volume_acceleration * volume_persistence * np.sign(data['close'] - data['open'])
    
    # Generate Final Factor Signal
    # Synthesize Divergence and Volume Components
    raw_signal = momentum_reversal_signal * volume_dynamics
    
    # Apply Volume-Based Signal Filter
    volume_percentile = data['volume'].rolling(15).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    volume_filter = (volume_percentile > volume_percentile.rolling(10).mean()).astype(int)
    
    # Final factor with volume filtering
    final_factor = raw_signal * volume_filter
    
    return final_factor
