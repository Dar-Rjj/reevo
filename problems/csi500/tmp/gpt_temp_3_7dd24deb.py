import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Gap Resolution Asymmetry Factor (C-GRAF)
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components with proper shifting
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_open'] = data['open'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Remove rows with NaN values from shifting
    data = data.dropna()
    
    # Core Gap Components
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['gap_to_range_ratio'] = np.abs(data['open'] - data['prev_close']) / data['daily_range']
    data['gap_utilization_efficiency'] = np.abs(data['open'] - data['prev_close']) / data['daily_range']
    
    # Resolution Efficiency Measurement
    data['price_movement'] = np.abs(data['close'] - data['open'])
    data['volume_per_gap_unit'] = data['volume'] / (data['price_movement'] + 1e-8)
    data['gap_efficiency'] = data['gap_to_range_ratio'] * data['volume_per_gap_unit']
    
    # Gap fill assessment
    gap_size = np.abs(data['open'] - data['prev_close'])
    data['gap_fill_percentage'] = np.abs(data['close'] - data['prev_close']) / (gap_size + 1e-8)
    data['fill_speed'] = np.abs(data['close'] - data['open']) / (gap_size + 1e-8)
    
    # Volatility Regime Context
    prev_range = data['prev_high'] - data['prev_low']
    data['range_compression_ratio'] = data['daily_range'] / (prev_range + 1e-8)
    data['regime_shift_magnitude'] = np.abs(data['range_compression_ratio'] - 1)
    
    # Volatility persistence
    prev_gap_momentum = (data['prev_open'] - data['close'].shift(2)) / (data['close'].shift(2) + 1e-8)
    data['volatility_persistence'] = data['gap_momentum'] - prev_gap_momentum
    
    # Liquidity Asymmetry Analysis
    data['volume_efficiency'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['directional_pressure'] = np.sign(data['close'] - data['open']) * data['volume']
    data['range_pressure_interaction'] = data['volume_efficiency'] * data['directional_pressure']
    data['volume_expansion'] = data['volume'] / (data['prev_volume'] + 1e-8)
    
    # Price Rejection & Efficiency Asymmetry
    data['upper_rejection'] = (data['high'] - data['close']) / (data['daily_range'] + 1e-8)
    data['lower_rejection'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['net_rejection_bias'] = data['upper_rejection'] - data['lower_rejection']
    
    # Efficiency Asymmetry Patterns
    data['volume_concentration_efficiency'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['price_impact'] = data['price_movement'] / (data['volume'] + 1e-8)
    
    # Amount-Volume Transition Dynamics
    data['amount_velocity'] = data['amount'] / (data['prev_amount'] + 1e-8)
    data['volume_velocity'] = data['volume'] / (data['prev_volume'] + 1e-8)
    data['transition_mismatch'] = (data['amount_velocity'] - data['volume_velocity']) * data['regime_shift_magnitude']
    
    data['amount_per_price_unit'] = data['amount'] / (data['price_movement'] + 1e-8)
    data['volume_per_price_unit'] = data['volume'] / (data['price_movement'] + 1e-8)
    data['transition_efficiency_ratio'] = data['amount_velocity'] / (data['volume_velocity'] + 1e-8)
    
    # Adaptive Composite Signal Construction
    
    # Gap Resolution Component
    gap_resolution = data['gap_efficiency'] * data['volume_per_gap_unit']
    
    # Volatility-Liquidity Component
    volatility_liquidity = data['range_pressure_interaction'] * data['regime_shift_magnitude']
    
    # Transition Asymmetry Component
    transition_asymmetry = data['transition_mismatch'] * data['volume_efficiency']
    
    # Final Cross-Sectional Factor
    # Combine components with appropriate weighting and interactions
    factor = (
        gap_resolution * volatility_liquidity * 
        np.sign(transition_asymmetry) * 
        data['net_rejection_bias'] * 
        data['gap_fill_percentage']
    )
    
    # Apply cross-sectional ranking normalization
    def cross_sectional_rank(series):
        return series.rank(pct=True) - 0.5
    
    # Create final factor with cross-sectional ranking
    final_factor = factor.groupby(data.index).transform(cross_sectional_rank)
    
    return final_factor
