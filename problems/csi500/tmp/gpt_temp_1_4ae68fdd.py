import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate previous close
    data['prev_close'] = data['close'].shift(1)
    
    # 1. Gap Strength Component
    # Opening Gap: (Open - Previous_Close) / Previous_Close
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Gap Persistence: 3-day same-sign gap ratio
    data['gap_sign'] = np.sign(data['opening_gap'])
    data['same_sign_count'] = 0
    for i in range(2, len(data)):
        current_sign = data['gap_sign'].iloc[i]
        prev_signs = [data['gap_sign'].iloc[i-1], data['gap_sign'].iloc[i-2]]
        same_sign_count = sum(1 for sign in prev_signs if sign == current_sign)
        data.loc[data.index[i], 'same_sign_count'] = same_sign_count
    
    data['gap_persistence'] = data['same_sign_count'] / 2.0  # Normalize to [0,1]
    
    # Combined Gap Strength
    data['gap_strength'] = data['opening_gap'].abs() * (1 + data['gap_persistence'])
    
    # 2. Intraday Momentum Component
    # Price Efficiency: (Close - (High + Low)/2) / (High - Low)
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['price_range'] = data['high'] - data['low']
    data['price_efficiency'] = np.where(
        data['price_range'] > 0,
        (data['close'] - data['mid_price']) / data['price_range'],
        0
    )
    
    # Volume Confirmation: Current Volume / 5-day Volume Average
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_confirmation'] = data['volume'] / data['volume_5d_avg']
    
    # Combined Intraday Momentum
    data['intraday_momentum'] = data['price_efficiency'] * data['volume_confirmation']
    
    # 3. Volatility Adjustment
    # 5-day Returns Volatility
    data['returns'] = data['close'].pct_change()
    data['volatility_5d'] = data['returns'].rolling(window=5, min_periods=1).std()
    
    # 4. Final Factor: Gap Strength × Intraday Momentum / Volatility Adjustment
    # Avoid division by zero and handle edge cases
    volatility_adj = np.where(data['volatility_5d'] > 0, data['volatility_5d'], 1e-6)
    data['factor'] = data['gap_strength'] * data['intraday_momentum'] / volatility_adj
    
    # Return the factor series with the same index as input
    return data['factor']
