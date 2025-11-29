import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Breakout Component
    # Calculate True Range
    data['Close_prev'] = data['close'].shift(1)
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['Close_prev']),
            np.abs(data['low'] - data['Close_prev'])
        )
    )
    data['TR_prev'] = data['TR'].shift(1)
    data['TR_prev2'] = data['TR'].shift(2)
    data['TR_acceleration'] = (data['TR'] - data['TR_prev']) - (data['TR_prev'] - data['TR_prev2'])
    
    # Identify Breakout Direction
    data['breakout_up'] = (data['high'] > data['open'] + (data['open'] - data['low'])).astype(int)
    data['breakout_down'] = (data['low'] < data['open'] - (data['high'] - data['open'])).astype(int)
    data['breakout_direction'] = data['breakout_up'] - data['breakout_down']
    
    # Intraday Reversal Component
    data['reversal_ratio'] = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)
    data['volatility_weight'] = data['close'].rolling(window=10, min_periods=5).std() / (data['close'] + 1e-8)
    data['reversal_component'] = data['reversal_ratio'] * data['volatility_weight']
    
    # Momentum Divergence Component
    data['high_close_momentum'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['low_close_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    data['High_3d'] = data['high'].rolling(window=3, min_periods=2).max()
    data['Low_3d'] = data['low'].rolling(window=3, min_periods=2).min()
    data['High_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['Low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    
    data['range_momentum'] = (data['High_3d'] - data['Low_3d']) / (data['High_5d'] - data['Low_5d'] + 1e-8) - 1
    data['momentum_divergence'] = (data['high_close_momentum'] - data['low_close_momentum']) * data['range_momentum']
    
    # Volume-Price Confirmation Component
    data['position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['liquidity_adjustment'] = data['amount'] / (data['volume'] + 1e-8)
    data['volume_confirmation'] = data['position_strength'] * data['volume_ratio'] * data['liquidity_adjustment']
    
    # Factor Integration
    data['vol_breakout_reversal'] = data['breakout_direction'] * data['reversal_component']
    data['combined_factor'] = data['vol_breakout_reversal'] * data['momentum_divergence'] * data['volume_confirmation']
    
    # Apply momentum persistence (weighted average with previous day)
    data['factor_persisted'] = 0.7 * data['combined_factor'] + 0.3 * data['combined_factor'].shift(1)
    
    # Cross-sectional rank
    data['factor_rank'] = data.groupby(data.index)['factor_persisted'].transform(lambda x: x.rank(pct=True))
    
    # Return the final factor values
    return data['factor_rank']
