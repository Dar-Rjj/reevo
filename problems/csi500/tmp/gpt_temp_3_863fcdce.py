import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['high_low_range'] = data['high'] - data['low']
    
    # Price Reversal-Momentum Component
    # Gap Reversal-Momentum: (Open - Previous Close) / (High - Low) × (High - Low)/Open
    data['gap_reversal_momentum'] = ((data['open'] - data['prev_close']) / data['high_low_range']) * (data['high_low_range'] / data['open'])
    
    # Intraday Mean Reversion: (High + Low)/2 - Close / (High - Low)
    data['intraday_mean_reversion'] = ((data['high'] + data['low']) / 2 - data['close']) / data['high_low_range']
    
    # Multi-day Return Decay: (Close - 5-day prior Close) - (Close - Previous Close)
    data['close_5d_prior'] = data['close'].shift(5)
    data['multi_day_return_decay'] = (data['close'] - data['close_5d_prior']) - (data['close'] - data['prev_close'])
    
    # Core Reversal-Momentum
    data['core_reversal_momentum'] = data['gap_reversal_momentum'] * data['intraday_mean_reversion'] * data['multi_day_return_decay']
    
    # Volume-Price Dynamics Component
    # Volume Spike Relative to Range: Volume / 5-day average of Volume × (High - Low) / 5-day average of (High - Low)
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['high_low_range_5d_avg'] = data['high_low_range'].rolling(window=5, min_periods=1).mean()
    data['volume_spike_relative_range'] = (data['volume'] / data['volume_5d_avg']) * (data['high_low_range'] / data['high_low_range_5d_avg'])
    
    # Price-Volume Divergence: sign(Close - Previous Close) × (Volume - 5-day average of Volume) / 5-day standard deviation of Volume
    data['volume_5d_std'] = data['volume'].rolling(window=5, min_periods=1).std()
    data['price_volume_divergence'] = np.sign(data['close'] - data['prev_close']) * (data['volume'] - data['volume_5d_avg']) / data['volume_5d_std']
    
    # Volume-Weighted Divergence: (High - Low)/Open × (Volume / 5-day rolling mean of Volume)
    data['volume_weighted_divergence'] = (data['high_low_range'] / data['open']) * (data['volume'] / data['volume_5d_avg'])
    
    # Volume-Enhanced Factor
    data['volume_enhanced_factor'] = (data['core_reversal_momentum'] * 
                                    data['volume_spike_relative_range'] * 
                                    data['price_volume_divergence'] * 
                                    data['volume_weighted_divergence'])
    
    # Liquidity-Microstructure Component
    # Liquidity Absorption: (High - Low) / Volume × 10000
    data['liquidity_absorption'] = (data['high_low_range'] / data['volume']) * 10000
    
    # Volatility Compression: 1 - (High - Low) / 10-day average of (High - Low)
    data['high_low_range_10d_avg'] = data['high_low_range'].rolling(window=10, min_periods=1).mean()
    data['volatility_compression'] = 1 - (data['high_low_range'] / data['high_low_range_10d_avg'])
    
    # Liquidity Score: (Close/((High + Low + Close)/3) - 1) × Volume
    data['liquidity_score'] = ((data['close'] / ((data['high'] + data['low'] + data['close']) / 3)) - 1) * data['volume']
    
    # Liquidity-Integrated Factor
    data['liquidity_integrated_factor'] = (data['volume_enhanced_factor'] * 
                                         data['liquidity_absorption'] * 
                                         data['volatility_compression'] * 
                                         data['liquidity_score'])
    
    # Momentum-Liquidity Interaction
    # Divergence-Liquidity: (High - Low)/Open × Liquidity Score
    data['divergence_liquidity'] = (data['high_low_range'] / data['open']) * data['liquidity_score']
    
    # Close Divergence-Volume: (High - Close)/Open × (Volume / 5-day rolling mean of Volume)
    data['close_divergence_volume'] = ((data['high'] - data['close']) / data['open']) * (data['volume'] / data['volume_5d_avg'])
    
    # Flow Ratio: (Volume when Close > Open) / (Volume when Close < Open)
    # Calculate using rolling windows for historical data only
    data['close_gt_open'] = data['close'] > data['open']
    data['close_lt_open'] = data['close'] < data['open']
    
    # Use expanding windows to ensure no future data
    volume_close_gt_open = []
    volume_close_lt_open = []
    
    for i in range(len(data)):
        if i == 0:
            volume_close_gt_open.append(np.nan)
            volume_close_lt_open.append(np.nan)
        else:
            # Use data up to current day only
            window_data = data.iloc[:i+1]
            vol_gt = window_data.loc[window_data['close_gt_open'], 'volume'].sum()
            vol_lt = window_data.loc[window_data['close_lt_open'], 'volume'].sum()
            volume_close_gt_open.append(vol_gt if vol_gt > 0 else 1)
            volume_close_lt_open.append(vol_lt if vol_lt > 0 else 1)
    
    data['volume_close_gt_open'] = volume_close_gt_open
    data['volume_close_lt_open'] = volume_close_lt_open
    data['flow_ratio'] = data['volume_close_gt_open'] / data['volume_close_lt_open']
    
    # Final Alpha Factor: 5-day rolling mean of (Divergence-Liquidity + Close Divergence-Volume + Liquidity-Integrated Factor × Flow Ratio)
    data['final_alpha_component'] = (data['divergence_liquidity'] + 
                                   data['close_divergence_volume'] + 
                                   data['liquidity_integrated_factor'] * data['flow_ratio'])
    
    # 5-day rolling mean for final factor
    final_factor = data['final_alpha_component'].rolling(window=5, min_periods=1).mean()
    
    return final_factor
