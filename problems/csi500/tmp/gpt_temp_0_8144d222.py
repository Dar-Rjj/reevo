import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Session Volatility Structure Dynamics factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic volatility measures
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['prev2_daily_range'] = data['daily_range'].shift(2)
    data['prev_close'] = data['close'].shift(1)
    
    # Estimate 30-minute ranges (assuming first/last 30 minutes)
    # For simplicity, we'll use opening and closing ranges as proxies
    data['open_range'] = data['high'] - data['low']  # Using daily range as proxy
    data['close_range'] = data['high'] - data['low']  # Using daily range as proxy
    
    # Estimate volume/amount for first/last 30 minutes (using daily totals as proxy)
    data['volume_first_30min'] = data['volume'] * 0.3  # Assuming 30% of volume in first 30min
    data['volume_last_30min'] = data['volume'] * 0.25  # Assuming 25% of volume in last 30min
    data['amount_first_30min'] = data['amount'] * 0.3
    data['amount_last_30min'] = data['amount'] * 0.25
    
    # Intraday Volatility Regime Patterns
    # Opening Volatility Structure
    opening_vol_expansion = ((data['open_range'] / data['prev_daily_range']) * 
                            np.sign(data['open'] - data['prev_close']))
    
    early_vol_compression = ((data['open_range'] / data['prev2_daily_range']) * 
                            ((data['close'] - data['open']) / data['open_range']))
    
    vol_structure_persistence = ((np.abs(data['open_range']) / np.abs(data['prev_daily_range'])) * 
                               ((data['close'] - data['open']) / data['open_range']))
    
    # Closing Volatility Regime
    late_vol_expansion = ((data['close_range'] / data['open_range']) * 
                         np.sign(data['close'] - data['open']))
    
    eod_vol_structure = ((data['close_range'] / data['prev_daily_range']) * 
                        ((data['close'] - data['open']) / data['close_range']))
    
    vol_regime_shift = (np.sign(data['close_range'] - data['open_range']) * 
                       np.sign(data['close'] - data['open']))
    
    # Volume-Volatility Dynamics
    # Volatility Volume Distribution
    morning_vol_volume = ((data['volume_first_30min'] / data['volume']) * 
                         (data['open_range'] / data['prev_daily_range']))
    
    afternoon_vol_volume = ((data['volume_last_30min'] / data['volume']) * 
                           (data['close_range'] / data['open_range']))
    
    volume_vol_divergence = morning_vol_volume - afternoon_vol_volume
    
    # Amount-Volatility Structure
    vol_amount_intensity = ((data['amount_first_30min'] / data['amount']) * 
                           (data['open_range'] / data['prev_daily_range']))
    
    amount_vol_persistence = ((data['amount_last_30min'] / data['amount']) * 
                             (data['close_range'] / data['open_range']))
    
    amount_vol_confirmation = (np.sign(data['open_range'] - data['prev_daily_range']) * 
                              (data['amount_first_30min'] / data['amount']))
    
    # Multi-session Volatility Patterns
    # Volatility Regime Persistence
    vol_direction = np.sign(data['daily_range'] - data['prev_daily_range'])
    prev_vol_direction = vol_direction.shift(1)
    
    consecutive_vol_days = vol_direction.rolling(window=5).apply(
        lambda x: np.sum(x == x.shift(1)), raw=False
    )
    
    vol_regime_ratio = ((data['daily_range'] / data['prev_daily_range']) * 
                       (data['open_range'] / data['prev2_daily_range']))
    
    vol_direction_consistency = (np.sign(data['daily_range'] - data['prev_daily_range']) * 
                                np.sign(data['close'] - data['open']))
    
    # Volatility Gap Integration
    vol_gap_magnitude = ((data['open_range'] / data['prev_daily_range']) * 
                        (data['close_range'] / data['open_range']))
    
    gap_vol_efficiency = (((data['close'] - np.minimum(data['open'], data['prev_close'])) / 
                          data['open_range']) * 
                         np.sign(data['open_range'] - data['prev_daily_range']))
    
    multi_day_vol_gap = ((data['daily_range'] / data['prev_daily_range']) * 
                        (data['open_range'] / data['prev2_daily_range']))
    
    # Combine factors with appropriate weights
    factor = (
        # Intraday patterns
        0.15 * opening_vol_expansion +
        0.12 * early_vol_compression +
        0.10 * vol_structure_persistence +
        0.13 * late_vol_expansion +
        0.11 * eod_vol_structure +
        0.09 * vol_regime_shift +
        
        # Volume dynamics
        0.08 * morning_vol_volume +
        0.07 * afternoon_vol_volume +
        0.06 * volume_vol_divergence +
        0.05 * vol_amount_intensity +
        0.04 * amount_vol_persistence +
        0.03 * amount_vol_confirmation +
        
        # Multi-session patterns
        0.02 * consecutive_vol_days +
        0.02 * vol_regime_ratio +
        0.01 * vol_direction_consistency +
        0.01 * vol_gap_magnitude +
        0.01 * gap_vol_efficiency +
        0.01 * multi_day_vol_gap
    )
    
    # Handle NaN values
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    return factor
