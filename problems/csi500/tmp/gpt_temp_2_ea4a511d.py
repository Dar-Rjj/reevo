import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Distribution & Boundary Dynamics factor
    Analyzes opening session momentum quality, boundary resistance patterns, 
    temporal momentum fragmentation, and momentum distribution quality
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day close
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate first hour metrics (assuming first hour is first trading period)
    # For simplicity, we'll use the first available data point as opening period
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.max())
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.min())
    data['first_hour_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.sum()/2)
    
    # Previous day volume
    data['prev_volume'] = data['volume'].shift(1)
    
    # 1. Opening Session Momentum Quality
    # Gap Momentum Efficiency
    gap = data['open'] - data['prev_close']
    volume_signal = np.sign(data['first_hour_volume'] - data['prev_volume'])
    gap_momentum_efficiency = gap * volume_signal
    
    # Opening Range Concentration
    opening_range = data['first_hour_high'] - data['first_hour_low']
    gap_abs = np.abs(data['open'] - data['prev_close'])
    opening_range_concentration = np.where(gap_abs > 0, opening_range / gap_abs, 0)
    
    # Morning Momentum Establishment
    first_hour_direction = np.sign(data['close'] - data['open'])  # Using close-open as proxy
    momentum_persistence = first_hour_direction * np.sign(data['close'] - data['open'])
    
    # 2. Boundary Momentum Resistance Patterns
    # Range Expansion Dynamics
    high_expansion = data['high'] - data['first_hour_high']
    low_expansion = data['first_hour_low'] - data['low']
    range_expansion_dynamics = high_expansion - low_expansion
    
    # Key Level Breakout Quality
    close_first_hour_diff = data['close'] - data['open']  # Using open as proxy for first_hour_close
    breakout_occurred = ((data['high'] != data['first_hour_high']) | 
                        (data['low'] != data['first_hour_low'])).astype(int)
    key_level_breakout_quality = close_first_hour_diff * breakout_occurred
    
    # 3. Temporal Momentum Fragmentation
    # Morning-Afternoon Momentum Divergence (using open-close as proxy)
    morning_momentum = np.sign(data['open'] - data['prev_close'])
    afternoon_momentum = np.sign(data['close'] - data['open'])
    session_momentum_divergence = morning_momentum * afternoon_momentum
    
    # Hourly Momentum Fragmentation (simplified using rolling windows)
    price_changes = data['close'].diff()
    direction_changes = (np.sign(price_changes) != np.sign(price_changes.shift(1))).astype(int)
    momentum_fragmentation = direction_changes.rolling(window=5, min_periods=1).sum()
    
    # 4. Momentum Distribution Quality Assessment
    # Price-Volume Elasticity Patterns
    price_move = np.abs(data['close'] - data['open'])
    volume_elasticity = np.where(data['volume'] > 0, price_move / data['volume'], 0)
    
    # Momentum Concentration vs Fragmentation
    daily_range = data['high'] - data['low']
    normalized_volatility = daily_range / data['close']
    momentum_stability = 1.0 / (1.0 + normalized_volatility)
    
    # Session Boundary Transitions
    opening_establishment = np.abs(data['open'] - data['prev_close']) / data['prev_close']
    closing_resolution = np.abs(data['close'] - data['open']) / data['open']
    boundary_transition_quality = opening_establishment * closing_resolution
    
    # Combine factors with appropriate weights
    factor = (
        0.25 * gap_momentum_efficiency +
        0.15 * opening_range_concentration +
        0.10 * momentum_persistence +
        0.15 * range_expansion_dynamics +
        0.10 * key_level_breakout_quality +
        0.10 * session_momentum_divergence +
        0.05 * momentum_fragmentation +
        0.05 * volume_elasticity +
        0.03 * momentum_stability +
        0.02 * boundary_transition_quality
    )
    
    # Handle NaN values
    factor = factor.fillna(0)
    
    return pd.Series(factor, index=data.index)
