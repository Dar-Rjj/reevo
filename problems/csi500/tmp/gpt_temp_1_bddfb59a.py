import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic metrics
    df['range'] = df['high'] - df['low']
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['price_change'] = df['close'].pct_change()
    
    # Cross-Session Price-Volume Fractal Momentum
    # Multi-session fractal patterns
    df['range_3d_ma'] = df['range'].rolling(window=3, min_periods=2).mean()
    df['volume_3d_ma'] = df['volume'].rolling(window=3, min_periods=2).mean()
    
    # Fractal expansion/compression signals
    df['range_expansion'] = (df['range'] > df['range_3d_ma']).astype(int)
    df['volume_expansion'] = (df['volume'] > df['volume_3d_ma']).astype(int)
    
    # Fractal alignment score
    df['fractal_alignment'] = (df['range_expansion'] == df['volume_expansion']).astype(int)
    
    # Fractal momentum (3-day persistence)
    df['fractal_momentum'] = df['fractal_alignment'].rolling(window=3, min_periods=2).sum()
    
    # Extreme Price Position Volume Confirmation
    # High/low position efficiency
    df['high_position'] = (df['close'] - df['low']) / df['range']
    df['low_position'] = (df['high'] - df['close']) / df['range']
    
    # Volume confirmation at extremes
    high_volume_threshold = df['volume'].rolling(window=5, min_periods=3).quantile(0.7)
    low_volume_threshold = df['volume'].rolling(window=5, min_periods=3).quantile(0.3)
    
    df['high_vol_confirmation'] = ((df['high_position'] > 0.7) & (df['volume'] > high_volume_threshold)).astype(int)
    df['low_vol_confirmation'] = ((df['low_position'] > 0.7) & (df['volume'] < low_volume_threshold)).astype(int)
    
    # Extreme position momentum
    df['extreme_momentum'] = df['high_vol_confirmation'] - df['low_vol_confirmation']
    
    # Intraday Amount Distribution Position Bias
    # Using open/close as proxy for intraday periods
    df['open_to_close'] = df['close'] - df['open']
    df['amount_ma'] = df['amount'].rolling(window=3, min_periods=2).mean()
    
    # Amount-position alignment
    df['amount_position_alignment'] = np.sign(df['open_to_close']) * np.sign(df['amount'] - df['amount_ma'])
    
    # Amount distribution bias (3-day persistence)
    df['amount_bias'] = (df['amount_position_alignment'] > 0).astype(int).rolling(window=3, min_periods=2).mean()
    
    # Range Boundary Volume Asymmetry Efficiency
    # Boundary efficiency metrics
    df['upper_boundary_efficiency'] = (df['high'] - df['close'].shift(1)) / df['range']
    df['lower_boundary_efficiency'] = (df['close'].shift(1) - df['low']) / df['range']
    
    # Volume asymmetry at boundaries
    df['upper_boundary_volume'] = ((df['upper_boundary_efficiency'] > 0.8) & (df['volume'] > df['volume_3d_ma'])).astype(int)
    df['lower_boundary_volume'] = ((df['lower_boundary_efficiency'] > 0.8) & (df['volume'] > df['volume_3d_ma'])).astype(int)
    
    df['boundary_asymmetry'] = df['upper_boundary_volume'] - df['lower_boundary_volume']
    
    # Price-Velocity Volume Acceleration Hierarchy
    # Multi-scale velocity
    df['price_velocity_1d'] = df['price_change'].abs()
    df['price_velocity_3d'] = df['close'].pct_change(periods=3).abs()
    
    df['volume_velocity_1d'] = df['volume'].pct_change().abs()
    df['volume_velocity_3d'] = df['volume'].pct_change(periods=3).abs()
    
    # Velocity-acceleration alignment
    df['velocity_alignment'] = (np.sign(df['price_velocity_1d'] - df['price_velocity_3d']) == 
                               np.sign(df['volume_velocity_1d'] - df['volume_velocity_3d'])).astype(int)
    
    # Hierarchy momentum (3-day persistence)
    df['hierarchy_momentum'] = df['velocity_alignment'].rolling(window=3, min_periods=2).mean()
    
    # Combine all factors with weights
    factors = [
        df['fractal_momentum'] * 0.25,
        df['extreme_momentum'] * 0.20,
        df['amount_bias'] * 0.20,
        df['boundary_asymmetry'] * 0.15,
        df['hierarchy_momentum'] * 0.20
    ]
    
    # Calculate final factor value
    for i, date in enumerate(df.index):
        if i < 5:  # Ensure enough data for rolling calculations
            result.loc[date] = 0
        else:
            factor_values = [f.loc[date] for f in factors if not pd.isna(f.loc[date])]
            if factor_values:
                result.loc[date] = np.nanmean(factor_values)
            else:
                result.loc[date] = 0
    
    # Fill any remaining NaN values
    result = result.fillna(0)
    
    return result
