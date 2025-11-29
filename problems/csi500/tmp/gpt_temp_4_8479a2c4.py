import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Price-Volume Asymmetry Factor
    Combines directional, volume, and microstructure asymmetry patterns
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data.groupby(level=1)['close'].shift(1)
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['is_up_day'] = (data['daily_return'] > 0).astype(int)
    data['is_down_day'] = (data['daily_return'] < 0).astype(int)
    
    # Directional Asymmetry Components
    # Bull Day Efficiency: (High - Open) / (Close - Low) on Up Days
    bull_efficiency = np.where(
        data['is_up_day'] == 1,
        (data['high'] - data['open']) / np.maximum(data['close'] - data['low'], 0.001),
        0
    )
    
    # Bear Day Resilience: (Open - Low) / (High - Close) on Down Days
    bear_resilience = np.where(
        data['is_down_day'] == 1,
        (data['open'] - data['low']) / np.maximum(data['high'] - data['close'], 0.001),
        0
    )
    
    # Asymmetry Ratio: Bull Day Efficiency / Bear Day Resilience
    asymmetry_ratio = np.where(
        (data['is_up_day'] == 1) | (data['is_down_day'] == 1),
        bull_efficiency / np.maximum(bear_resilience, 0.001),
        1
    )
    
    # Gap Asymmetry Dynamics
    gap_magnitude = np.abs(data['open'] - data['prev_close'])
    gap_direction = np.sign(data['open'] - data['prev_close'])
    
    # Gap Opening Persistence: (Close - Open) / |Open - Previous Close|
    gap_persistence = np.where(
        gap_magnitude > 0,
        (data['close'] - data['open']) / gap_magnitude,
        0
    )
    
    # Gap Filling Efficiency: (High - Low) / |Open - Previous Close|
    gap_filling = np.where(
        gap_magnitude > 0,
        (data['high'] - data['low']) / gap_magnitude,
        0
    )
    
    # Volume-Price Asymmetry Assessment
    # Use amount and volume to estimate trade size distribution
    avg_trade_size = data['amount'] / np.maximum(data['volume'], 1)
    
    # Volume skew pattern approximation using daily price movement
    volume_skew = np.where(
        data['is_up_day'] == 1,
        data['volume'],
        -data['volume']
    )
    
    # Normalize volume skew by total volume
    volume_skew_normalized = volume_skew / np.maximum(data['volume'], 1)
    
    # Microstructure Asymmetry Patterns
    mid_price = (data['high'] + data['low']) / 2
    
    # Opening strength vs closing weakness
    opening_strength = (data['open'] - data['prev_close']) / np.maximum(data['prev_close'], 0.001)
    closing_weakness = (data['close'] - mid_price) / np.maximum(mid_price, 0.001)
    microstructure_asymmetry = opening_strength - closing_weakness
    
    # Price level asymmetry - resistance vs support efficiency
    daily_range = data['high'] - data['low']
    high_proximity = (data['high'] - data['close']) / np.maximum(daily_range, 0.001)
    low_proximity = (data['close'] - data['low']) / np.maximum(daily_range, 0.001)
    price_level_asymmetry = high_proximity - low_proximity
    
    # Combine asymmetry components with appropriate weights
    directional_component = asymmetry_ratio * gap_persistence
    volume_component = volume_skew_normalized * avg_trade_size
    microstructure_component = microstructure_asymmetry * price_level_asymmetry
    
    # Final asymmetry factor
    asymmetry_factor = (
        0.4 * directional_component +
        0.35 * volume_component +
        0.25 * microstructure_component
    )
    
    # Apply 1-day lag to ensure no forward-looking bias
    result = data.groupby(level=1).apply(
        lambda x: x['asymmetry_factor'].shift(1)
        if 'asymmetry_factor' in x.columns else pd.Series(index=x.index)
    )
    
    # Handle multi-index structure
    if isinstance(result, pd.Series) and result.index.nlevels == 2:
        result = result.droplevel(0)
    
    return result
