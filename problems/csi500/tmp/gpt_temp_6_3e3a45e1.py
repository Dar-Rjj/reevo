import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Expansion Detection & Breakout Ratio
    # Current Intraday Volatility Components
    data['high_low_range'] = data['high'] - data['low']
    data['open_close_gap'] = data['open'] - data['close'].shift(1)
    data['open_close_gap'] = data['open_close_gap'].fillna(0)
    
    # Volatility Expansion Ratio
    data['range_ma5'] = data['high_low_range'].rolling(window=5, min_periods=1).mean()
    data['vol_expansion_ratio'] = data['high_low_range'] / data['range_ma5']
    data['vol_expansion_ratio'] = data['vol_expansion_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Breakout Direction Strength
    data['breakout_ratio'] = (data['close'] - data['low']) / (data['high_low_range'] + 1e-8)
    data['breakout_ratio'] = data['breakout_ratio'].clip(0, 1)
    data['vol_adjusted_breakout'] = data['breakout_ratio'] * data['vol_expansion_ratio']
    
    # Volume-Regime Acceleration System
    # Volume Momentum Assessment
    data['volume_ma10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma10'] + 1e-8)
    
    # Volume Momentum: Volume Change Rate
    data['volume_momentum'] = data['volume'].pct_change(periods=1).fillna(0)
    
    # Volume-Regime Application
    data['volume_accelerated_breakout'] = np.where(
        data['volume_ratio'] > 1.2,
        data['vol_adjusted_breakout'],
        data['vol_adjusted_breakout'] * data['volume_ratio']
    )
    
    # Momentum Efficiency & Range Expansion
    # Momentum Efficiency Components
    data['price_momentum'] = data['close'] - data['open']
    data['efficiency_ratio'] = (data['close'] - data['open']) / (data['high_low_range'] + 1e-8)
    data['efficiency_ratio'] = data['efficiency_ratio'].clip(-1, 1)
    
    # Range Expansion Weighting
    data['range_expansion'] = data['high_low_range'] / (data['close'].shift(1) + 1e-8)
    data['momentum_weighted_breakout'] = data['volume_accelerated_breakout'] * data['range_expansion']
    
    # Concentration & Trend Alignment
    # Concentration Confirmation
    data['concentration_level'] = data['amount'] / (data['volume'] + 1e-8)
    data['concentration_weighted_factor'] = data['momentum_weighted_breakout'] * data['concentration_level']
    
    # Trend-Breakout Alignment
    data['price_slope'] = data['close'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    data['trend_direction'] = np.sign(data['price_slope'])
    data['trend_direction'] = data['trend_direction'].replace(0, 1)
    
    # Final Composite Factor
    data['final_factor'] = data['concentration_weighted_factor'] * data['trend_direction']
    
    return data['final_factor']
