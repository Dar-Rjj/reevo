import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap-Volatility Dynamics
    data['overnight_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volatility_compression'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3))
    data['gap_volatility_interaction'] = data['overnight_gap_momentum'] * data['volatility_compression']
    
    # Position Efficiency Analysis
    data['movement_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['position_divergence'] = ((data['close'] - data['low']) - (data['open'] - data['low'])) / (data['high'] - data['low'])
    data['efficiency_position_alignment'] = data['movement_efficiency'] * data['position_divergence']
    
    # Volume-Momentum Integration
    data['volume_volatility_ratio'] = data['amount'] / (data['high'] - data['low'])
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['volume_momentum'] = data['gap_momentum'] * data['volume_volatility_ratio']
    
    # Multi-timeframe Breakout Signals
    # Short-term Breakout
    data['volatility_breakout'] = data['volatility_compression'] * data['gap_momentum']
    data['position_breakout'] = data['position_divergence'] * data['movement_efficiency']
    
    # Medium-term Context
    high_5d = data['high'].rolling(window=6, min_periods=6).max()
    low_5d = data['low'].rolling(window=6, min_periods=6).min()
    data['price_trend'] = (data['close'] - data['close'].shift(5)) / (high_5d - low_5d)
    
    # Volume persistence: count consecutive days with Amount_t > Amount_{t-1}
    volume_increase = (data['amount'] > data['amount'].shift(1)).astype(int)
    data['volume_persistence'] = volume_increase.groupby(volume_increase.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if (x == 1).any() else 0, raw=False
    ).reset_index(level=0, drop=True)
    
    # Cross-timeframe Alignment
    data['trend_breakout'] = data['volatility_breakout'] * data['price_trend']
    data['volume_persistence_breakout'] = data['position_breakout'] * data['volume_persistence']
    
    # Final Alpha Construction
    # Core Breakout Components
    data['gap_volatility_breakout'] = data['gap_volatility_interaction'] * data['volume_momentum']
    data['efficiency_position_breakout'] = data['efficiency_position_alignment'] * data['volume_volatility_ratio']
    data['multi_timeframe_breakout'] = data['trend_breakout'] * data['volume_persistence_breakout']
    
    # Signal Refinement
    data['breakout_strength'] = data['gap_volatility_breakout'] * data['efficiency_position_breakout']
    data['timing_enhancement'] = data['breakout_strength'] * data['multi_timeframe_breakout']
    
    # Risk-Aware Final Signal
    data['position_extremes'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * \
                               (1 - ((data['close'] - data['low']) / (data['high'] - data['low'])))
    
    # Final Alpha
    data['alpha'] = data['timing_enhancement'] * data['position_extremes']
    
    # Return the alpha factor series
    return data['alpha']
