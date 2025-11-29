import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price changes and volume metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['price_change'] = data['close'] - data['open']
    data['abs_price_change'] = abs(data['price_change'])
    data['volume_change'] = data['volume'].pct_change()
    data['avg_volume'] = data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Intraday Price Structure
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['session_structure_ratio'] = (data['high'] - data['open']) / (data['close'] - data['low'] + 1e-8)
    data['price_compression'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8)
    data['intraday_reversal'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['close'] - data['open'])
    
    # Volume Distribution Analysis (simplified - assuming no intraday data)
    data['volume_skew'] = 1.0  # Placeholder - would require intraday data
    data['volume_concentration'] = 1.0  # Placeholder - would require intraday data
    data['volume_persistence'] = np.sign(data['volume_change']) * np.sign(data['price_change'])
    data['volume_efficiency'] = data['abs_price_change'] / (data['volume'] / (data['avg_volume'] + 1e-8) + 1e-8)
    
    # Price-Volume Interaction
    data['price_change_rank'] = data['price_change'].rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank().iloc[-1])
    data['volume_change_rank'] = data['volume_change'].rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank().iloc[-1])
    data['volume_price_divergence'] = data['price_change_rank'] - data['volume_change_rank']
    data['breakout_volume'] = (data['high'] - data['prev_high']) * data['volume']
    data['support_volume'] = (data['low'] - data['prev_low']) * data['volume']
    data['volume_weighted_range'] = (data['high'] - data['low']) * (data['volume'] / (data['avg_volume'] + 1e-8))
    
    # Session-Based Integration
    data['opening_strength'] = data['opening_gap'] * data['volume_skew']
    data['intraday_momentum'] = data['price_change'] * data['volume_persistence']
    data['range_efficiency'] = data['abs_price_change'] / (data['price_compression'] + 1e-8)
    data['volume_confirmation'] = data['price_change'] * data['volume_price_divergence']
    
    # Composite Factor Construction
    data['core_momentum'] = data['intraday_momentum'] * data['range_efficiency']
    data['volume_validation'] = data['core_momentum'] * data['volume_confirmation']
    data['opening_signal'] = data['volume_validation'] * data['opening_strength']
    data['final_factor'] = data['opening_signal'] * data['intraday_reversal']
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
