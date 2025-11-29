import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate midpoint
    data['midpoint'] = (data['high'] + data['low']) / 2
    
    # 1. Measure Intraday Trend Strength
    # Midpoint movement from open
    data['midpoint_movement'] = data['midpoint'] - data['open']
    
    # Trend consistency - compare high/low proximity to close
    data['high_proximity'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['low_proximity'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Directional bias ratio
    data['directional_bias'] = np.where(
        data['midpoint_movement'] > 0,
        data['low_proximity'],  # For uptrends, focus on low proximity
        data['high_proximity']   # For downtrends, focus on high proximity
    )
    
    # Trend strength score
    data['trend_strength'] = (
        np.abs(data['midpoint_movement'] / (data['open'] + 1e-8)) * 
        data['directional_bias']
    )
    
    # 2. Evaluate Liquidity Acceleration
    # Volume velocity - current volume relative to rolling median
    data['volume_median_5d'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['volume_velocity'] = data['volume'] / (data['volume_median_5d'] + 1e-8) - 1
    
    # Volume acceleration - change in volume velocity
    data['volume_acceleration'] = data['volume_velocity'].diff()
    
    # Liquidity momentum - smoothed acceleration
    data['liquidity_momentum'] = data['volume_acceleration'].rolling(window=3, min_periods=2).mean()
    
    # Liquidity trend persistence
    data['liquidity_persistence'] = (
        data['volume_velocity'].rolling(window=5, min_periods=3).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0
        )
    )
    
    # Liquidity acceleration score
    data['liquidity_acceleration'] = (
        data['volume_acceleration'] * 
        (1 + np.abs(data['liquidity_persistence']))
    )
    
    # 3. Combine Trend and Liquidity Signals
    # Raw factor
    data['raw_factor'] = data['trend_strength'] * data['liquidity_acceleration']
    
    # Directional weighting
    directional_weight = np.sign(data['midpoint_movement']) * (1 + np.abs(data['liquidity_momentum']))
    
    # Final factor
    data['factor'] = data['raw_factor'] * directional_weight
    
    # Return the factor series
    return data['factor']
