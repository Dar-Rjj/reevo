import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate basic price components
    data['momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['prev_momentum'] = data['momentum'].shift(1)
    data['momentum_acceleration'] = data['momentum'] - data['prev_momentum']
    
    # Price level momentum
    data['high_breakout_momentum'] = (data['close'] - data['high']) / (data['high'] - data['low']).replace(0, np.nan)
    data['low_breakdown_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Momentum persistence
    data['price_change'] = data['close'] - data['open']
    data['consecutive_days'] = 0
    for i in range(1, len(data)):
        if (data['price_change'].iloc[i] > 0 and data['price_change'].iloc[i-1] > 0) or \
           (data['price_change'].iloc[i] < 0 and data['price_change'].iloc[i-1] < 0):
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
    
    data['momentum_magnitude_change'] = abs(data['momentum']) - abs(data['prev_momentum'])
    
    # Volume components
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_2days_ago'] = data['volume'].shift(2)
    data['volume_change'] = data['volume'] / data['prev_volume'].replace(0, np.nan)
    data['prev_volume_change'] = data['prev_volume'] / data['volume_2days_ago'].replace(0, np.nan)
    data['volume_acceleration'] = data['volume_change'] - data['prev_volume_change']
    
    # Volume-price synchronization
    data['volume_weighted_price_change'] = (data['close'] - data['open']) * data['volume']
    data['volume_price_correlation'] = np.sign(data['close'] - data['open']) * np.sign(data['volume_change'] - 1)
    
    # Volume regime detection
    data['volume_roll_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_roll_quantile'] = data['volume'].rolling(window=10, min_periods=5).quantile(0.75)
    data['high_volume_regime'] = (data['volume'] > data['volume_roll_quantile']).astype(int)
    data['volume_breakout'] = data['volume'] / data['volume_roll_avg'].replace(0, np.nan)
    
    # Volatility regime switching
    data['daily_range'] = data['high'] - data['low']
    data['range_roll_quantile_80'] = data['daily_range'].rolling(window=5, min_periods=3).quantile(0.8)
    data['range_roll_quantile_20'] = data['daily_range'].rolling(window=5, min_periods=3).quantile(0.2)
    data['prev_range'] = data['daily_range'].shift(1)
    data['volatility_expansion'] = data['daily_range'] / data['prev_range'].replace(0, np.nan)
    
    data['high_volatility'] = (data['daily_range'] > data['range_roll_quantile_80']).astype(int)
    data['low_volatility'] = (data['daily_range'] < data['range_roll_quantile_20']).astype(int)
    
    # Volatility-adjusted momentum
    data['range_normalized_momentum'] = (data['close'] - data['open']) / data['daily_range'].replace(0, np.nan)
    data['prev_close'] = data['close'].shift(1)
    data['volatility_scaled_returns'] = (data['close'] - data['prev_close']) / data['daily_range'].replace(0, np.nan)
    
    # Regime-dependent weighting
    data['volatility_multiplier'] = 1.0
    data.loc[data['high_volatility'] == 1, 'volatility_multiplier'] = 0.7
    data.loc[data['low_volatility'] == 1, 'volatility_multiplier'] = 1.3
    data.loc[data['volatility_expansion'] > 1.5, 'volatility_multiplier'] = data['volatility_multiplier'] * 1.2
    
    # Price-level anchoring effects
    data['daily_range_position'] = (data['close'] - data['low']) / data['daily_range'].replace(0, np.nan)
    data['prev_range_position'] = data['daily_range_position'].shift(1)
    
    # Price level reversion
    data['extreme_high_reversion'] = np.where(
        data['daily_range_position'] > 0.8,
        data['high'] - data['close'],
        0
    )
    data['extreme_low_reversion'] = np.where(
        data['daily_range_position'] < 0.2,
        data['close'] - data['low'],
        0
    )
    
    # Anchoring strength (simplified)
    data['price_cluster_density'] = abs(data['close'] - data['prev_close']) / data['daily_range'].replace(0, np.nan)
    data['anchoring_strength'] = 1 - data['price_cluster_density']
    
    # Composite alpha construction
    # Core momentum component
    data['core_momentum'] = data['momentum_acceleration'] * data['volume_price_correlation']
    
    # Volatility-regime adjusted signal
    data['volatility_adjusted_signal'] = data['core_momentum'] * data['volatility_multiplier']
    
    # Price-level anchoring adjustment
    data['price_reversion_factor'] = np.where(
        data['extreme_high_reversion'] != 0,
        -data['extreme_high_reversion'] / data['daily_range'].replace(0, np.nan),
        np.where(
            data['extreme_low_reversion'] != 0,
            data['extreme_low_reversion'] / data['daily_range'].replace(0, np.nan),
            data['anchoring_strength']
        )
    )
    
    data['anchoring_adjusted_signal'] = data['volatility_adjusted_signal'] * data['price_reversion_factor']
    
    # Volume acceleration confirmation
    data['final_signal'] = data['anchoring_adjusted_signal'] * data['volume_acceleration'] * data['volume_breakout']
    
    # Return the final factor values
    return data['final_signal']
