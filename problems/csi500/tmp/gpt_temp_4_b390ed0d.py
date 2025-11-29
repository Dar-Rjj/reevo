import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    data['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Compute Volatility-Adjusted Price Change
    data['vol_adj_price_change'] = ((data['close'] - data['open']) / data['true_range'].replace(0, np.nan)) * abs(data['close'] - data['open'])
    data['vol_adj_price_change'] = data['vol_adj_price_change'].fillna(0)
    
    # Calculate Price Position within Daily Range
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)
    data['price_position'] = (data['close'] - data['low']) / daily_range
    data['price_position'] = data['price_position'].fillna(0.5)
    
    # Calculate price position change for reversal detection
    data['position_change'] = data['price_position'] - data['price_position'].shift(1)
    data['position_change'] = data['position_change'].fillna(0)
    
    # Compute Pattern Persistence with Reversal Weighting
    data['direction'] = np.sign(data['vol_adj_price_change'])
    data['consecutive_days'] = 0
    for i in range(1, len(data)):
        if data['direction'].iloc[i] == data['direction'].iloc[i-1]:
            data['consecutive_days'].iloc[i] = data['consecutive_days'].iloc[i-1] + 1
        else:
            data['consecutive_days'].iloc[i] = 1
    
    # Apply reversal penalty for extended trends
    data['persistence_score'] = data['consecutive_days'] * abs(data['vol_adj_price_change'])
    reversal_penalty = np.where(data['consecutive_days'] > 3, 1 / (1 + 0.2 * (data['consecutive_days'] - 3)), 1)
    data['persistence_score'] = data['persistence_score'] * reversal_penalty
    
    # Calculate Volume-Based Confirmation Signals
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5'].replace(0, np.nan)
    data['volume_ratio'] = data['volume_ratio'].fillna(1)
    
    # Volume breakout using 20-day range
    data['volume_high_20'] = data['volume'].rolling(window=20, min_periods=1).max()
    data['volume_low_20'] = data['volume'].rolling(window=20, min_periods=1).min()
    data['volume_range'] = data['volume_high_20'] - data['volume_low_20']
    data['volume_range'] = data['volume_range'].replace(0, np.nan)
    data['volume_breakout'] = (data['volume'] - data['volume_low_20']) / data['volume_range']
    data['volume_breakout'] = data['volume_breakout'].fillna(0.5)
    
    # Compute Volume-Price Consistency (5-day correlation)
    data['abs_return'] = abs(data['close'] - data['close'].shift(1))
    volume_price_corr = []
    for i in range(len(data)):
        if i < 4:
            volume_price_corr.append(0)
        else:
            window_volume = data['volume'].iloc[i-4:i+1]
            window_abs_return = data['abs_return'].iloc[i-4:i+1]
            if window_volume.std() > 0 and window_abs_return.std() > 0:
                corr = np.corrcoef(window_volume, window_abs_return)[0,1]
                volume_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_price_corr.append(0)
    data['volume_price_corr'] = volume_price_corr
    data['volume_consistency'] = data['volume_price_corr'] * data['volume_ma_5']
    
    # Amount-Based Liquidity Enhancement
    data['amount_ma_10'] = data['amount'].rolling(window=10, min_periods=1).mean()
    data['amount_ratio'] = data['amount'] / data['amount_ma_10'].replace(0, np.nan)
    data['amount_ratio'] = data['amount_ratio'].fillna(1)
    
    # Combine volume and amount liquidity measures
    data['liquidity_score'] = (data['volume_ratio'] + data['amount_ratio']) / 2
    data['liquidity_multiplier'] = np.where(data['liquidity_score'] > 1.2, 1.2, 
                                          np.where(data['liquidity_score'] < 0.8, 0.8, data['liquidity_score']))
    
    # Incorporate gap effects
    data['overnight_gap'] = (data['open'] - prev_close) / prev_close.replace(0, np.nan)
    data['overnight_gap'] = data['overnight_gap'].fillna(0)
    data['gap_persistence'] = 1 / (1 + abs(data['overnight_gap']) * 10)
    
    # Combine Volatility-Adjusted Pattern and Reversal Components
    data['pattern_signal'] = data['vol_adj_price_change'] * data['persistence_score']
    data['reversal_weight'] = 1 + abs(data['position_change']) * 2
    data['combined_signal'] = data['pattern_signal'] * data['reversal_weight'] * data['gap_persistence']
    
    # Apply Liquidity Confirmation
    data['volume_breakout_multiplier'] = np.where(data['volume_breakout'] > 0.7, 1.2, 
                                                np.where(data['volume_breakout'] < 0.3, 0.8, 1.0))
    data['final_signal'] = (data['combined_signal'] * 
                          data['volume_breakout_multiplier'] * 
                          (1 + data['volume_consistency'] / data['volume_consistency'].abs().mean()) * 
                          data['liquidity_multiplier'])
    
    # Apply Mean-Reversion Logic with Volatility Scaling
    data['tr_ma_5'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['tr_ratio'] = data['true_range'] / data['tr_ma_5'].replace(0, np.nan)
    data['tr_ratio'] = data['tr_ratio'].fillna(1)
    
    # Volatility regime adjustment
    volatility_weight = np.where(data['tr_ratio'] > 1.5, 1.5, 
                               np.where(data['tr_ratio'] < 0.7, 0.7, data['tr_ratio']))
    
    # Gap-induced volatility adjustment
    gap_vol_adjust = 1 + abs(data['overnight_gap']) * 5
    
    # Final factor with directional reversal and volatility scaling
    data['factor'] = -data['final_signal'] * volatility_weight * gap_vol_adjust
    
    return data['factor']
