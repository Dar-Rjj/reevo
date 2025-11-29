import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Pressure Divergence Factor
    # Opening Pressure
    opening_gap = (data['open'] / data['close'].shift(1) - 1).abs()
    opening_pressure = opening_gap
    
    # Closing Pressure
    close_to_high_distance = (data['high'] - data['close']) / data['high']
    closing_pressure = close_to_high_distance
    
    # Pressure Ratio and Momentum
    pressure_ratio = closing_pressure / (opening_pressure + 1e-8)
    pressure_ratio_3d_ago = pressure_ratio.shift(3)
    pressure_momentum = pressure_ratio - pressure_ratio_3d_ago
    
    # Amount Momentum
    amount_pct_change = data['amount'].pct_change(3)
    amount_momentum_sign = np.sign(amount_pct_change)
    
    # Intraday Pressure Divergence
    intraday_pressure_div = pressure_momentum * amount_momentum_sign
    
    # Volume-Weighted Range Divergence
    # Volume-Weighted Range
    daily_range = (data['high'] - data['low']) / data['close']
    volume_weighted_range = daily_range * data['volume']
    
    # Range-Volume Momentum
    range_volume_current = volume_weighted_range
    range_volume_7d_ago = volume_weighted_range.shift(7)
    range_volume_momentum = range_volume_current - range_volume_7d_ago
    
    # Price Momentum
    price_pct_change = data['close'].pct_change(7)
    price_momentum_sign = np.sign(price_pct_change)
    
    # Volume-Weighted Range Divergence
    volume_range_div = range_volume_momentum * price_momentum_sign
    
    # Opening-Closing Divergence Factor
    # Opening Strength
    opening_return = (data['open'] / data['close'].shift(1) - 1)
    
    # Closing Strength
    closing_return = (data['close'] / data['open'] - 1)
    
    # Opening-Closing Divergence
    oc_divergence = opening_return - closing_return
    oc_divergence_4d_ago = oc_divergence.shift(4)
    oc_divergence_momentum = oc_divergence - oc_divergence_4d_ago
    
    # Volume Pattern
    volume_variance = data['volume'].rolling(window=4, min_periods=1).var()
    volume_pattern_sign = np.sign(volume_variance.pct_change())
    
    # Opening-Closing Divergence Factor
    opening_closing_div = oc_divergence_momentum * volume_pattern_sign
    
    # High-Low Pressure Divergence
    # High Pressure
    high_to_open_distance = (data['high'] - data['open']) / data['open']
    high_pressure = high_to_open_distance
    
    # Low Pressure
    low_to_open_distance = (data['low'] - data['open']) / data['open']
    low_pressure = low_to_open_distance.abs()
    
    # Pressure Asymmetry
    pressure_asymmetry = high_pressure / (low_pressure + 1e-8)
    pressure_asymmetry_6d_ago = pressure_asymmetry.shift(6)
    asymmetry_momentum = pressure_asymmetry - pressure_asymmetry_6d_ago
    
    # Amount Flow
    amount_change = data['amount'].pct_change(6)
    amount_momentum_sign_hl = np.sign(amount_change)
    
    # High-Low Pressure Divergence
    high_low_pressure_div = asymmetry_momentum * amount_momentum_sign_hl
    
    # Volume-Range Efficiency Divergence
    # Range Efficiency
    price_range = data['high'] - data['low']
    volume_per_unit_range = data['volume'] / (price_range + 1e-8)
    range_efficiency = volume_per_unit_range
    
    # Efficiency Momentum
    efficiency_current = range_efficiency
    efficiency_5d_ago = range_efficiency.shift(5)
    efficiency_momentum = efficiency_current - efficiency_5d_ago
    
    # Price Direction Consistency
    returns_5d = data['close'].pct_change()
    sign_consistency = returns_5d.rolling(window=5, min_periods=1).apply(
        lambda x: 1 if (x > 0).sum() > (x < 0).sum() else -1 if (x < 0).sum() > (x > 0).sum() else 0
    )
    price_pattern_sign = sign_consistency
    
    # Volume-Range Efficiency Divergence
    volume_range_efficiency_div = efficiency_momentum * price_pattern_sign
    
    # Combine all factors with equal weighting
    combined_factor = (
        intraday_pressure_div.rank(pct=True) +
        volume_range_div.rank(pct=True) +
        opening_closing_div.rank(pct=True) +
        high_low_pressure_div.rank(pct=True) +
        volume_range_efficiency_div.rank(pct=True)
    ) / 5
    
    return combined_factor
