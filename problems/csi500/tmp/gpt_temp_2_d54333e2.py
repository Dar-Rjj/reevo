import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # 1. Momentum-Volatility Elasticity
    # Price efficiency
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume intensity
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_intensity'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    # Momentum acceleration
    data['momentum_1d'] = data['close'] / data['prev_close'] - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_acceleration'] = data['momentum_1d'] / (np.abs(data['momentum_3d']) + 1e-8)
    
    # 2. Gap Momentum Integration
    # Overnight gap efficiency
    data['overnight_gap'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    
    # Gap preservation
    data['gap_preservation'] = (data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    
    # Gap volume alignment
    data['gap_volume_alignment'] = data['volume'] / (data['prev_volume'] + 1e-8)
    
    # 3. Range-Constrained Patterns
    # Extreme position
    data['extreme_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Range expansion
    data['range_expansion'] = (data['high'] - data['low']) / (data['prev_close'] + 1e-8)
    
    # Session persistence (using midday price as average of open and close)
    data['midday_price'] = (data['open'] + data['close']) / 2
    data['session_persistence'] = (data['close'] - data['midday_price']) / (data['high'] - data['low'] + 1e-8)
    
    # 4. Volume-Volatility Alignment
    # Volume concentration
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Volume momentum
    data['volume_momentum'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    # Volume efficiency
    data['volume_efficiency'] = data['volume'] / (np.abs(data['close'] - data['open']) + 1e-8)
    
    # 5. Breakout & Reversion Assessment
    # Breakout efficiency
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['breakout_efficiency'] = data['daily_range'] / (data['range_5d_avg'] + 1e-8)
    
    # Reversion momentum (using momentum following extreme positions)
    data['extreme_position_prev'] = data['extreme_position'].shift(1)
    data['reversion_momentum'] = np.where(
        (data['extreme_position_prev'] > 0.8) | (data['extreme_position_prev'] < 0.2),
        data['momentum_1d'],
        0
    )
    
    # 6. Composite Elasticity Factor
    # Calculate cross-sectional z-scores for each component
    components = [
        'price_efficiency', 'volume_intensity', 'momentum_acceleration',
        'overnight_gap', 'gap_preservation', 'gap_volume_alignment',
        'extreme_position', 'range_expansion', 'session_persistence',
        'volume_concentration', 'volume_momentum', 'volume_efficiency',
        'breakout_efficiency', 'reversion_momentum'
    ]
    
    # Calculate cross-sectional z-scores
    for col in components:
        data[f'{col}_zscore'] = data.groupby(data.index)[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    # Define positive regime components (aligned momentum-volatility with volume confirmation)
    positive_components = [
        'price_efficiency_zscore', 'volume_intensity_zscore', 'momentum_acceleration_zscore',
        'gap_preservation_zscore', 'gap_volume_alignment_zscore', 'volume_momentum_zscore'
    ]
    
    # Define negative pattern components (diverging relationships without volume support)
    negative_components = [
        'overnight_gap_zscore', 'range_expansion_zscore', 'volume_concentration_zscore',
        'breakout_efficiency_zscore', 'reversion_momentum_zscore'
    ]
    
    # Calculate composite scores
    data['positive_regime'] = data[positive_components].mean(axis=1)
    data['negative_patterns'] = data[negative_components].mean(axis=1)
    
    # Final cross-sectional elasticity factor
    data['elasticity_factor'] = data['positive_regime'] - 0.5 * data['negative_patterns']
    
    # Apply volatility regime weighting using range expansion as proxy
    volatility_regime = np.where(data['range_expansion'] > data['range_expansion'].rolling(window=20).median(), 1.2, 0.8)
    data['weighted_elasticity'] = data['elasticity_factor'] * volatility_regime
    
    # Return the final factor series
    return data['weighted_elasticity']
