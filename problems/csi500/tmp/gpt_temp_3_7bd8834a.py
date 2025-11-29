import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Micro-Gap Velocity Efficiency Synthesis factor
    Combines gap dynamics, volume validation, range regimes, and volume-price asymmetry
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price changes and gaps
    data['prev_close'] = data['close'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['gap'] = data['open'] - data['prev_close']
    data['gap_pct'] = data['gap'] / data['prev_close']
    
    # Gap Velocity Momentum
    data['amount_ratio'] = data['amount'] / data['prev_amount']
    data['gap_velocity'] = data['gap'] * data['amount_ratio']
    
    # Micro-Breakout Strength adjustment
    data['intraday_range'] = data['high'] - data['low']
    data['breakout_strength'] = np.where(
        data['gap'] > 0,
        (data['high'] - data['open']) / data['intraday_range'],
        (data['open'] - data['low']) / data['intraday_range']
    )
    data['adjusted_gap_velocity'] = data['gap_velocity'] * (1 + data['breakout_strength'])
    
    # Volume-Validated Gap Dynamics
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5']
    data['price_change'] = data['close'] - data['open']
    data['volume_gap_dynamics'] = data['volume_ratio'] * data['intraday_range'] * data['price_change']
    
    # Range-Regime Context
    data['range_ma_5'] = data['intraday_range'].rolling(window=5, min_periods=1).mean()
    data['range_regime'] = data['intraday_range'] / data['range_ma_5']
    data['range_enhanced_momentum'] = np.where(
        data['range_regime'] > 1,
        data['adjusted_gap_velocity'] * data['range_regime'],
        data['adjusted_gap_velocity'] * (2 - data['range_regime'])  # Contracting range penalty
    )
    
    # Intraday Timing - Gap absorption and fade
    data['gap_absorption'] = np.where(
        data['gap'] > 0,
        (data['close'] - data['open']) / data['gap'],  # Positive gap absorption
        (data['open'] - data['close']) / data['gap']   # Negative gap absorption
    )
    data['gap_fade'] = 1 - abs(data['gap_absorption'])
    
    # Gap Memory Effects
    data['gap_fill_distance'] = np.where(
        data['gap'] > 0,
        (data['low'] - data['prev_close']) / data['gap'],  # Positive gap fill progress
        (data['prev_close'] - data['high']) / data['gap']   # Negative gap fill progress
    )
    data['gap_memory'] = data['gap_fill_distance'] * data['volume_ratio']
    
    # Volume-Price Asymmetry
    data['directional_volume'] = np.where(
        data['price_change'] > 0,
        data['volume'] * (1 + data['price_change'] / data['open']),
        data['volume'] * (1 - data['price_change'] / data['open'])
    )
    data['volume_asymmetry'] = (data['directional_volume'] - data['volume_ma_5']) / data['volume_ma_5']
    
    # Final factor synthesis
    factor = (
        0.3 * data['range_enhanced_momentum'] +
        0.25 * data['volume_gap_dynamics'] +
        0.2 * data['gap_fade'] * np.sign(data['gap']) +
        0.15 * data['gap_memory'] +
        0.1 * data['volume_asymmetry']
    )
    
    # Normalize the factor
    factor_series = factor.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return factor_series
