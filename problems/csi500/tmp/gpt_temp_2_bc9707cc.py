import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Efficiency & Reversal Detection
    # Raw Price Movement Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['morning_return'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    
    # Intraday Reversal Patterns
    data['afternoon_return'] = (data['close'] - data['low']) / (data['low'] + 1e-8)
    data['intraday_reversal_signal'] = data['morning_return'] * data['afternoon_return']
    
    # Session Persistence Analysis
    opening_gap = data['open'] - data['close'].shift(1)
    data['gap_persistence'] = (data['close'] - data['open']) / (np.abs(opening_gap) + 1e-8)
    
    midday_price = (data['high'] + data['low']) / 2
    data['midday_persistence'] = (data['close'] - midday_price) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-Timeframe Acceleration Divergence
    # Ultra-Short Acceleration
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['acceleration_divergence'] = data['return_1d'] / (data['return_3d'] + 1e-8)
    
    # Short-Term Momentum Divergence
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 1e-8)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / (data['close'].shift(10) + 1e-8)
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_10d']
    
    # Volume-Intensity Confirmation System
    # Volume Intensity Metrics
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Calculate 5-day volume moving average
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma_5d'] + 1e-8)
    
    # Volume Persistence Analysis
    # Count consecutive days with volume above average
    above_avg_volume = data['volume'] > data['volume_ma_5d']
    data['volume_persistence'] = above_avg_volume.astype(int)
    data['volume_persistence'] = data['volume_persistence'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    data['volume_persistence_sqrt'] = np.sqrt(data['volume_persistence'])
    
    # Combined Factor Synthesis
    # Primary Acceleration-Reversal Component
    primary_component = (data['intraday_reversal_signal'] * data['price_efficiency'] * 
                        data['acceleration_divergence'] * data['gap_persistence'])
    
    # Volume-Weighted Confirmation
    volume_confirmation = data['volume_intensity'] * data['volume_persistence_sqrt']
    
    # Detect abnormal volume-price relationships
    high_efficiency_low_volume = (data['price_efficiency'] > data['price_efficiency'].rolling(window=10).mean()) & \
                                (data['volume_ratio'] < 1)
    low_efficiency_high_volume = (data['price_efficiency'] < data['price_efficiency'].rolling(window=10).mean()) & \
                                (data['volume_ratio'] > 1)
    
    divergence_strength = np.where(high_efficiency_low_volume, -1, 
                                  np.where(low_efficiency_high_volume, 1, 0))
    
    # Final Factor Construction
    final_factor = (primary_component * volume_confirmation * 
                   data['momentum_divergence'] * (1 + 0.1 * divergence_strength))
    
    return final_factor
