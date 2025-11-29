import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Price-Volume Divergence with Momentum Persistence alpha factor
    """
    data = df.copy()
    
    # Price Momentum Persistence Analysis
    # Intraday Momentum Strength
    data['morning_momentum'] = (data['high'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['low']) / data['low']
    data['full_day_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Momentum Consistency Across Timeframes
    data['momentum_same_direction'] = (
        (data['morning_momentum'] > 0) & (data['afternoon_momentum'] > 0) & (data['full_day_momentum'] > 0) |
        (data['morning_momentum'] < 0) & (data['afternoon_momentum'] < 0) & (data['full_day_momentum'] < 0)
    ).astype(int)
    
    data['momentum_mixed_direction'] = (
        ((data['morning_momentum'] > 0) & (data['afternoon_momentum'] > 0) & (data['full_day_momentum'] < 0)) |
        ((data['morning_momentum'] > 0) & (data['afternoon_momentum'] < 0) & (data['full_day_momentum'] > 0)) |
        ((data['morning_momentum'] < 0) & (data['afternoon_momentum'] > 0) & (data['full_day_momentum'] > 0)) |
        ((data['morning_momentum'] > 0) & (data['afternoon_momentum'] < 0) & (data['full_day_momentum'] < 0)) |
        ((data['morning_momentum'] < 0) & (data['afternoon_momentum'] > 0) & (data['full_day_momentum'] < 0)) |
        ((data['morning_momentum'] < 0) & (data['afternoon_momentum'] < 0) & (data['full_day_momentum'] > 0))
    ).astype(int)
    
    # Momentum Acceleration
    data['momentum_acceleration'] = data['afternoon_momentum'] / (data['morning_momentum'] + 1e-8)
    data['intraday_momentum_ratio'] = data['afternoon_momentum'] / (data['morning_momentum'] + 1e-8)
    
    # Momentum Persistence Score (using 5-day rolling window)
    data['momentum_direction'] = np.sign(data['full_day_momentum'])
    momentum_persistence = []
    for i in range(len(data)):
        if i < 5:
            momentum_persistence.append(1)
        else:
            current_dir = data['momentum_direction'].iloc[i]
            window = data['momentum_direction'].iloc[max(0, i-4):i+1]
            persistence_count = 0
            for j in range(len(window)-1, -1, -1):
                if window.iloc[j] == current_dir:
                    persistence_count += 1
                else:
                    break
            momentum_persistence.append(persistence_count)
    data['momentum_persistence_score'] = momentum_persistence
    
    # Volume-Price Divergence Detection
    # Since we only have total daily volume, we'll use proxies for intraday volume patterns
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_pattern_change'] = (data['volume'] - data['volume_5d_avg']) / (data['volume_5d_avg'] + 1e-8)
    
    # Price-Volume Divergence Signals
    data['high_price_low_volume'] = data['morning_momentum'] - data['volume_pattern_change']
    data['low_price_high_volume'] = -data['morning_momentum'] - data['volume_pattern_change']
    data['closing_divergence'] = data['full_day_momentum'] - data['volume_pattern_change']
    
    # Volume Pattern Persistence
    data['volume_5d_pattern'] = data['volume'].rolling(window=5, min_periods=1).std() / (data['volume'].rolling(window=5, min_periods=1).mean() + 1e-8)
    data['volume_pattern_stability'] = 1 / (1 + abs(data['volume_pattern_change']))
    
    # Price Range Efficiency Analysis
    data['daily_range'] = data['high'] - data['low']
    data['upper_range_efficiency'] = (data['high'] - data['open']) / (data['daily_range'] + 1e-8)
    data['lower_range_efficiency'] = (data['open'] - data['low']) / (data['daily_range'] + 1e-8)
    data['total_range_efficiency'] = abs(data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Range Compression vs Expansion
    data['prev_day_range'] = data['daily_range'].shift(1)
    data['daily_range_ratio'] = data['daily_range'] / (data['prev_day_range'] + 1e-8)
    
    # Range persistence calculation
    range_expansion_persistence = []
    range_compression_persistence = []
    for i in range(len(data)):
        if i < 5:
            range_expansion_persistence.append(0)
            range_compression_persistence.append(0)
        else:
            expansion_count = 0
            compression_count = 0
            window = data['daily_range_ratio'].iloc[max(0, i-4):i+1]
            for j in range(len(window)-1, -1, -1):
                if window.iloc[j] > 1.2:
                    expansion_count += 1
                else:
                    break
            for j in range(len(window)-1, -1, -1):
                if window.iloc[j] < 0.8:
                    compression_count += 1
                else:
                    break
            range_expansion_persistence.append(expansion_count)
            range_compression_persistence.append(compression_count)
    
    data['range_expansion_persistence'] = range_expansion_persistence
    data['range_compression_persistence'] = range_compression_persistence
    
    # Price Efficiency vs Volume
    data['high_efficiency_low_volume'] = data['total_range_efficiency'] - data['volume_pattern_change']
    data['low_efficiency_high_volume'] = -data['total_range_efficiency'] - data['volume_pattern_change']
    
    # Multi-Timeframe Signal Integration
    # Momentum-Volume Alignment Scoring
    data['momentum_volume_alignment'] = (
        data['momentum_same_direction'] * 1.0 + 
        data['momentum_mixed_direction'] * 0.5 +
        data['volume_pattern_stability'] * 0.5
    )
    
    # Range Efficiency Weighting
    efficiency_weight = np.where(
        data['total_range_efficiency'] > 0.7, 1.2,
        np.where(data['total_range_efficiency'] < 0.3, 0.8, 1.0)
    )
    
    # Final Alpha Factor Generation
    alpha_factor = (
        data['momentum_persistence_score'] * 
        data['momentum_volume_alignment'] * 
        efficiency_weight * 
        np.sign(data['full_day_momentum'])
    )
    
    # Clean up and return
    result = pd.Series(alpha_factor, index=data.index, name='intraday_price_volume_divergence')
    return result
