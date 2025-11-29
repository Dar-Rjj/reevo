import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure proper sorting by date
    data = data.sort_index()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_close_2'] = data['close'].shift(2)
    data['prev_close_3'] = data['close'].shift(3)
    data['prev_close_4'] = data['close'].shift(4)
    
    # Intraday Price Patterns
    data['opening_gap_momentum'] = (data['close'] - data['open']) * (data['open'] - data['prev_close'])
    data['midday_price_acceleration'] = ((data['high'] + data['low']) / 2) - ((data['open'] + data['close']) / 2)
    data['closing_strength_ratio'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * (data['close'] - data['open'])
    
    # Volume-Price Dynamics
    data['volume_weighted_price_efficiency'] = ((data['close'] - data['low']) * data['volume']) - ((data['high'] - data['close']) * data['volume'])
    data['price_range_efficiency'] = (data['high'] - data['low']) / (data['close'] * data['volume'] + 1e-8)
    data['volume_impact_ratio'] = (data['volume'] / (data['high'] - data['low'] + 1e-8)) * np.abs(data['close'] - data['open'])
    
    # Multi-period Momentum
    data['momentum_acceleration'] = (data['close'] - data['prev_close_2']) - (data['prev_close_2'] - data['prev_close_4'])
    
    # Calculate rolling high-low range for volatility adjustment
    data['rolling_high_3'] = data['high'].rolling(window=3, min_periods=1).max()
    data['rolling_low_3'] = data['low'].rolling(window=3, min_periods=1).min()
    data['volatility_adjusted_return'] = (data['close'] - data['prev_close_3']) / (data['rolling_high_3'] - data['rolling_low_3'] + 1e-8)
    
    data['gap_continuation'] = (data['open'] - data['prev_close']) * (data['close'] - data['prev_close'])
    
    # Composite Signals
    data['momentum_volume_composite'] = data['opening_gap_momentum'] * data['volume_weighted_price_efficiency']
    data['acceleration_efficiency_interaction'] = data['momentum_acceleration'] * data['price_range_efficiency']
    data['multi_timeframe_confirmation'] = data['gap_continuation'] * data['volatility_adjusted_return']
    
    # Combine all factors with equal weights
    factors = [
        'opening_gap_momentum',
        'midday_price_acceleration', 
        'closing_strength_ratio',
        'volume_weighted_price_efficiency',
        'price_range_efficiency',
        'volume_impact_ratio',
        'momentum_acceleration',
        'volatility_adjusted_return',
        'gap_continuation',
        'momentum_volume_composite',
        'acceleration_efficiency_interaction',
        'multi_timeframe_confirmation'
    ]
    
    # Calculate final factor as average of normalized individual factors
    result = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        day_data = data.loc[date]
        valid_factors = []
        
        for factor in factors:
            if pd.notna(day_data[factor]):
                valid_factors.append(day_data[factor])
        
        if valid_factors:
            # Cross-sectional normalization (z-score for the day)
            factor_array = np.array(valid_factors)
            if len(factor_array) > 1 and factor_array.std() > 0:
                normalized_factors = (factor_array - factor_array.mean()) / factor_array.std()
                result.loc[date] = normalized_factors.mean()
            else:
                result.loc[date] = 0
        else:
            result.loc[date] = 0
    
    return result
