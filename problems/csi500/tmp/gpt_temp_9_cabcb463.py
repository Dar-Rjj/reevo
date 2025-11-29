import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['price_elasticity_ratio'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 0.001)
    df['gap'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    
    # Short-term Elasticity-Fractal Momentum
    df['gap_change'] = df['gap'] - df['gap'].shift(1)
    df['range_ratio'] = df['range'] / (df['range'].shift(1) + 0.001)
    df['gap_fractal_momentum'] = df['gap_change'] * df['range_ratio']
    df['short_term_momentum'] = df['price_elasticity_ratio'] * df['gap_fractal_momentum']
    
    # Medium-term Elasticity-Fractal Decay
    df['gap_5_10'] = (df['gap'].shift(5) - df['gap'].shift(10))
    df['range_5_10'] = df['range'].shift(5) / (df['range'].shift(10) + 0.001)
    df['gap_fractal_decay'] = df['gap_5_10'] * df['range_5_10']
    df['elasticity_persistence'] = df['price_elasticity_ratio'].rolling(window=5).mean()
    df['medium_term_decay'] = df['elasticity_persistence'] * df['gap_fractal_decay']
    
    # Long-term Elasticity-Fractal Structure
    df['gap_20_40'] = (df['gap'].shift(20) - df['gap'].shift(40))
    df['range_20_40'] = df['range'].shift(20) / (df['range'].shift(40) + 0.001)
    df['structural_fractal'] = df['gap_20_40'] * df['range_20_40']
    df['elasticity_regime'] = df['price_elasticity_ratio'].rolling(window=20).std()
    df['long_term_structure'] = df['elasticity_regime'] * df['structural_fractal']
    
    # Elasticity-Volume Fractal Interaction
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_fractal_acceleration'] = df['volume_change'] * df['range_ratio']
    df['elasticity_volume_coupling'] = df['price_elasticity_ratio'] * (df['volume'] / (df['volume'].shift(1) + 0.001))
    df['volume_acceleration'] = df['elasticity_volume_coupling'] * df['volume_fractal_acceleration']
    
    # Volume Fractal Deceleration
    df['volume_5_10'] = (df['volume'].shift(5) - df['volume'].shift(10))
    df['range_5_10_vol'] = df['range'].shift(5) / (df['range'].shift(10) + 0.001)
    df['volume_fractal_deceleration'] = df['volume_5_10'] * df['range_5_10_vol']
    df['absorption_exhaustion'] = df['volume'].rolling(window=5).mean() / (df['volume'].rolling(window=10).mean() + 0.001)
    df['volume_deceleration'] = df['absorption_exhaustion'] * df['volume_fractal_deceleration']
    
    # Price-Volume Fractal Efficiency
    df['price_volume_fractal_efficiency'] = df['volume'] * df['gap'] / ((df['range'] + 0.001) ** 2)
    df['elasticity_volume_correlation'] = df['price_elasticity_ratio'].rolling(window=5).corr(df['volume'].rolling(window=5).mean())
    df['price_volume_efficiency'] = df['elasticity_volume_correlation'] * df['price_volume_fractal_efficiency']
    
    # Cross-Sectional Elasticity-Fractal Divergence
    df['volume_ratio'] = df['volume'] / (df['volume'].shift(1) + 0.001) - 1
    df['gap_fractal_divergence'] = df['gap'] * df['volume_ratio'] * df['range_ratio']
    df['absorption_elasticity_divergence'] = abs(df['price_elasticity_ratio'] - df['price_elasticity_ratio'].shift(1))
    df['cross_sectional_divergence'] = df['absorption_elasticity_divergence'] * df['gap_fractal_divergence']
    
    # Microstructure Elasticity Divergence
    df['microstructure_divergence'] = abs(df['gap']) * abs(df['volume_ratio']) * df['range']
    df['friction_elasticity_interaction'] = df['price_elasticity_ratio'] * df['volume_ratio']
    df['microstructure_elasticity'] = df['friction_elasticity_interaction'] * df['microstructure_divergence']
    
    # Elasticity-Fractal Volatility Regimes
    df['range_5d_mean'] = df['range'].rolling(window=5).mean()
    df['gap_ratio'] = df['gap'] / (abs(df['gap'].shift(1)) + 0.001)
    df['gap_fractal_volatility'] = (df['range'] / (df['range_5d_mean'] + 0.001)) * df['gap_ratio']
    df['friction_intensity'] = df['price_elasticity_ratio'].rolling(window=5).std()
    df['volatility_regime'] = df['friction_intensity'] * df['gap_fractal_volatility']
    
    # Elasticity-Fractal Efficiency
    df['gap_fractal_efficiency'] = (df['gap'] ** 2) / ((df['range'] + 0.001) ** 2) * df['volume_ratio']
    df['elasticity_friction_feedback'] = df['price_elasticity_ratio'] * df['volume_ratio']
    df['efficiency_factor'] = df['elasticity_friction_feedback'] * df['gap_fractal_efficiency']
    
    # Multi-timeframe integration with weights
    short_term_weight = 0.4
    medium_term_weight = 0.35
    long_term_weight = 0.25
    
    # Calculate weighted composite factor
    for date in df.index:
        if pd.notna(df.loc[date, 'short_term_momentum']) and pd.notna(df.loc[date, 'medium_term_decay']) and pd.notna(df.loc[date, 'long_term_structure']):
            momentum_component = (
                short_term_weight * df.loc[date, 'short_term_momentum'] +
                medium_term_weight * df.loc[date, 'medium_term_decay'] +
                long_term_weight * df.loc[date, 'long_term_structure']
            )
            
            volume_component = (
                0.4 * df.loc[date, 'volume_acceleration'] +
                0.3 * df.loc[date, 'volume_deceleration'] +
                0.3 * df.loc[date, 'price_volume_efficiency']
            )
            
            divergence_component = (
                0.6 * df.loc[date, 'cross_sectional_divergence'] +
                0.4 * df.loc[date, 'microstructure_elasticity']
            )
            
            regime_component = (
                0.5 * df.loc[date, 'volatility_regime'] +
                0.5 * df.loc[date, 'efficiency_factor']
            )
            
            # Final alpha signal
            result.loc[date] = (
                0.35 * momentum_component +
                0.25 * volume_component +
                0.25 * divergence_component +
                0.15 * regime_component
            )
        else:
            result.loc[date] = np.nan
    
    return result
