import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Compression Dynamics
    data['high_low_range'] = data['high'] - data['low']
    data['prev_high_low_range'] = data['high_low_range'].shift(1)
    data['intraday_volatility_collapse'] = data['high_low_range'] / data['prev_high_low_range']
    
    data['close_open_diff'] = abs(data['close'] - data['open'])
    data['price_elasticity_ratio'] = data['close_open_diff'] / data['high_low_range']
    
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_compression_signal'] = data['volume'] / data['prev_volume']
    
    # Price Elasticity Integration
    data['prev_close'] = data['close'].shift(1)
    data['opening_elasticity'] = (data['open'] - data['prev_close']) / data['prev_high_low_range']
    
    data['closing_elasticity'] = (data['close'] - data['open']) / data['high_low_range']
    
    data['midday_pressure'] = (data['high'] + data['low']) / 2 - data['open']
    
    # Compression-Elasticity Divergence
    data['volatility_breakout_signal'] = data['intraday_volatility_collapse'] * data['price_elasticity_ratio']
    
    data['elasticity_imbalance'] = data['opening_elasticity'] - data['closing_elasticity']
    
    data['compression_quality'] = data['volatility_breakout_signal'] * data['elasticity_imbalance']
    
    # Cross-Sectional Dynamics
    data['amount_elasticity'] = data['amount'] / (abs(data['close'] - data['open']) + 1e-8)
    
    # Session Transition - using volume from current day only (simplified)
    # Since we don't have intraday data, we'll use a proxy based on price movement
    data['session_transition'] = data['volume_compression_signal']
    
    # Alpha Factor Synthesis
    data['core_integration'] = data['compression_quality'] * data['midday_pressure']
    
    data['enhanced_elasticity'] = data['core_integration'] * data['amount_elasticity']
    
    # Cross-sectional ranking
    data['cross_sectional_alpha'] = data.groupby(data.index)['enhanced_elasticity'].transform(
        lambda x: x.rank(pct=True)
    ) * data['session_transition']
    
    # Handle infinite values and NaN
    data['cross_sectional_alpha'] = data['cross_sectional_alpha'].replace([np.inf, -np.inf], np.nan)
    
    return data['cross_sectional_alpha']
