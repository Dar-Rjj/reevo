import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Context and Breakout Detection
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling Average True Range (5-day)
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Breakout Strength Assessment
    data['morning_breakout'] = (data['high'] - data['open']) / data['open']
    data['afternoon_support'] = (data['close'] - data['low']) / data['close']
    
    # Historical Breakout Context
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['range_5d'] = data['high_5d'] - data['low_5d']
    
    data['breakout_ratio_high'] = (data['high'] - data['high_5d']) / np.where(data['range_5d'] == 0, 1, data['range_5d'])
    data['breakout_ratio_low'] = (data['low'] - data['low_5d']) / np.where(data['range_5d'] == 0, 1, data['range_5d'])
    
    # Volatility Persistence
    data['vol_persistence'] = data['close'].rolling(window=5, min_periods=3).std() / data['close'].rolling(window=10, min_periods=5).std()
    
    # Momentum Synthesis and Volatility Adjustment
    # Intraday Trend Components
    data['intraday_trend'] = (data['close'] - data['open']) / np.where((data['high'] - data['low']) == 0, 1, (data['high'] - data['low']))
    data['open_close_return'] = (data['close'] - data['open']) / data['open']
    data['prev_close_open_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Volatility-Adjusted Momentum
    data['vol_adj_intraday_trend'] = data['intraday_trend'] / np.where(data['atr_5'] == 0, 1, data['atr_5'])
    data['vol_adj_return'] = data['open_close_return'] / np.where(data['atr_5'] == 0, 1, data['atr_5'])
    
    # Momentum Direction and Strength
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['momentum_direction'] = np.where(data['close'] > data['midpoint'], 1, -1)
    data['vol_adj_momentum_dir'] = data['momentum_direction'] / np.where(data['true_range'] == 0, 1, data['true_range'])
    
    # Volume-Price Divergence and Multi-Timeframe Alignment
    # Volume Analysis
    data['volume_5d_mean'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / np.where(data['volume_5d_mean'] == 0, 1, data['volume_5d_mean'])
    
    # Volume Acceleration Factor
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_acceleration'] = data['volume'] / np.where(data['prev_volume'] == 0, 1, data['prev_volume'])
    data['volume_growth_3d'] = data['volume'].rolling(window=3, min_periods=2).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
    
    # Amount Efficiency
    data['amount_efficiency'] = data['amount'] / np.where((data['high'] - data['low']) == 0, 1, (data['high'] - data['low']))
    
    # Volume-Price Divergence Detection
    data['volume_ratio_deviation'] = data['volume_ratio'] - 1
    data['volume_price_divergence'] = np.sign(data['intraday_trend']) * data['volume_ratio_deviation']
    data['return_volume_divergence'] = np.sign(data['open_close_return']) * data['volume_ratio_deviation']
    
    # Multi-Timeframe Volume Confirmation
    data['volume_10d_mean'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_regime'] = data['volume_5d_mean'] / np.where(data['volume_10d_mean'] == 0, 1, data['volume_10d_mean'])
    
    data['volume_momentum_confirmation'] = np.where(
        np.sign(data['volume_acceleration']) == np.sign(data['momentum_direction']), 1, -1
    )
    
    # Price Position and Range Context Integration
    # Price Position Components
    data['close_5d_mean'] = data['close'].rolling(window=5, min_periods=3).mean()
    data['relative_price'] = (data['close'] - data['close_5d_mean']) / np.where((data['high'] - data['low']) == 0, 1, (data['high'] - data['low']))
    
    data['price_deviation'] = data['close'] - data['close_5d_mean']
    data['vol_adj_position'] = data['price_deviation'] / np.where(data['atr_5'] == 0, 1, data['atr_5'])
    
    # Range Momentum and Alignment
    data['daily_range'] = data['high'] - data['low']
    data['range_3d_ago'] = data['daily_range'].shift(3)
    data['range_momentum'] = (data['daily_range'] - data['range_3d_ago']) / np.where(data['range_3d_ago'] == 0, 1, data['range_3d_ago'])
    
    data['close_3d_ago'] = data['close'].shift(3)
    data['price_momentum_3d'] = (data['close'] - data['close_3d_ago']) / np.where(data['close_3d_ago'] == 0, 1, data['close_3d_ago'])
    data['momentum_alignment'] = np.sign(data['price_momentum_3d']) * np.sign(data['range_momentum'])
    
    # Breakout Persistence Context
    data['breakout_strength'] = (data['breakout_ratio_high'] + data['morning_breakout']) / 2
    data['breakout_persistence'] = data['breakout_strength'].rolling(window=10, min_periods=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] > x.iloc[i-1]]) / max(len(x)-1, 1)
    )
    
    # Final Alpha Construction
    # Core Momentum Component
    data['core_momentum'] = data['vol_adj_momentum_dir'] * data['intraday_trend']
    
    # Volume-Price Alignment Enhancement
    data['volume_enhancement'] = data['volume_price_divergence'] * data['volume_momentum_confirmation']
    data['enhanced_momentum'] = data['core_momentum'] * data['volume_enhancement']
    
    # Position and Range Context Integration
    data['position_range_integration'] = data['vol_adj_position'] * data['range_momentum']
    data['integrated_component'] = data['enhanced_momentum'] * data['position_range_integration']
    
    # Regime and Persistence Finalization
    data['final_alpha'] = (
        data['integrated_component'] * 
        data['vol_persistence'] * 
        data['breakout_persistence'] * 
        data['momentum_alignment']
    )
    
    # Clean up intermediate columns
    result = data['final_alpha'].copy()
    
    return result
