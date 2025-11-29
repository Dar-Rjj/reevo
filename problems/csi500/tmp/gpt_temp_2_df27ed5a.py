import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Gap Fracture Momentum & Volume Absorption Dynamics factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = data['open'] - data['prev_close']
    data['gap_pct'] = data['gap'] / data['prev_close']
    data['intraday_range'] = data['high'] - data['low']
    data['abs_gap'] = abs(data['gap'])
    
    # Gap-Induced Momentum Fractures
    # Gap size vs intraday range ratio
    data['gap_range_ratio'] = data['abs_gap'] / (data['intraday_range'] + 1e-8)
    
    # Gap filling smoothness breakdown
    data['gap_fill_path'] = np.where(
        data['gap'] > 0,
        (data['low'] - data['prev_close']) / (data['gap'] + 1e-8),  # Down gap fill progress
        (data['high'] - data['prev_close']) / (data['gap'] + 1e-8)   # Up gap fill progress
    )
    data['gap_fill_jaggedness'] = data['gap_fill_path'].rolling(window=5).std()
    
    # Gap micro-structure noise
    data['gap_oscillations'] = (data['high'] - data['low']) / (data['abs_gap'] + 1e-8)
    
    # Volume-Gap Decoupling
    data['volume_ma'] = data['volume'].rolling(window=10).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1e-8)
    data['gap_volume_decoupling'] = data['gap_pct'] * (1 - np.tanh(data['volume_ratio']))
    
    # Gap momentum persistence
    data['gap_momentum'] = np.where(
        data['gap'] > 0,
        (data['close'] - data['open']) / (data['gap'] + 1e-8),
        (data['open'] - data['close']) / (data['gap'] + 1e-8)
    )
    data['gap_momentum_persistence'] = data['gap_momentum'].rolling(window=3).mean()
    
    # Volume Absorption Analysis
    # Gap volume-weighted pressure
    data['gap_volume_pressure'] = data['gap_pct'] * data['volume_ratio']
    
    # Volume absorption at gap levels
    data['volume_absorption'] = np.where(
        data['gap'] > 0,
        (data['close'] - data['low']) / (data['intraday_range'] + 1e-8) * data['volume_ratio'],
        (data['high'] - data['close']) / (data['intraday_range'] + 1e-8) * data['volume_ratio']
    )
    
    # Gap market depth proxy using amount
    data['amount_ma'] = data['amount'].rolling(window=10).mean()
    data['amount_ratio'] = data['amount'] / (data['amount_ma'] + 1e-8)
    data['gap_market_depth'] = data['gap_pct'] * data['amount_ratio']
    
    # Price Path Efficiency with Gap Fracture Integration
    # Gap fracture-adjusted efficiency
    data['return_efficiency'] = (data['close'] - data['open']) / (data['abs_gap'] + 1e-8)
    
    # Gap fracture recovery momentum
    data['fracture_recovery'] = (data['close'] - data['open']) / (data['intraday_range'] + 1e-8)
    
    # Gap fracture resistance strength
    data['gap_resistance'] = np.where(
        data['gap'] > 0,
        (data['high'] - data['prev_close']) / (data['abs_gap'] + 1e-8),
        (data['prev_close'] - data['low']) / (data['abs_gap'] + 1e-8)
    )
    
    # Volume-Gap Fracture Alignment
    data['volume_fracture_alignment'] = data['gap_momentum'] * data['volume_absorption']
    
    # Volume absorption-weighted efficiency
    data['efficiency_volume_weighted'] = data['return_efficiency'] * data['volume_absorption']
    
    # Composite Alpha Factor Generation
    # Combine Gap Fracture with Volume Absorption
    gap_fracture_strength = (
        data['gap_range_ratio'] * 
        data['gap_fill_jaggedness'] * 
        data['gap_volume_decoupling']
    )
    
    volume_absorption_efficiency = (
        data['volume_absorption'] * 
        data['gap_market_depth'] * 
        data['volume_fracture_alignment']
    )
    
    # Main composite factor
    composite_factor = gap_fracture_strength * volume_absorption_efficiency
    
    # Apply efficiency confirmation filter
    efficiency_filter = data['efficiency_volume_weighted'] * data['fracture_recovery']
    filtered_factor = composite_factor * efficiency_filter
    
    # Multi-timeframe validation
    short_term_ma = filtered_factor.rolling(window=5).mean()
    medium_term_ma = filtered_factor.rolling(window=10).mean()
    
    # Cross-sectional normalization
    final_factor = filtered_factor / (filtered_factor.rolling(window=20).std() + 1e-8)
    
    # Remove extreme outliers
    factor_quantile = final_factor.rolling(window=50).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-8)
    )
    
    # Final factor with consistency validation
    validated_factor = final_factor * np.where(
        (short_term_ma > medium_term_ma) & (factor_quantile.between(0.1, 0.9)),
        1.0, 0.5  # Reduce weight for inconsistent signals
    )
    
    return validated_factor
