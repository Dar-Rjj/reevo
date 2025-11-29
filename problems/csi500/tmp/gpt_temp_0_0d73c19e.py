import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Breakout-Rejection Component
    # Calculate Breakout Momentum
    data['rolling_high_max'] = data['high'].rolling(window=5, min_periods=3).max()
    data['rolling_low_min'] = data['low'].rolling(window=5, min_periods=3).min()
    data['high_breakout'] = (data['high'] / data['rolling_high_max']) - 1
    data['low_breakout'] = 1 - (data['low'] / data['rolling_low_min'])
    
    # Calculate Intraday Price Rejection
    data['upper_rejection'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_rejection'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['net_rejection'] = data['upper_rejection'] - data['lower_rejection']
    
    # Volatility-Volume Compression Component
    # Volatility Compression
    # Compute True Range
    high_low = data['high'] - data['low']
    high_prev_close = abs(data['high'] - data['close'].shift(1))
    low_prev_close = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Volatility Compression Ratio
    data['tr_rolling_median'] = data['true_range'].rolling(window=5, min_periods=3).median()
    data['volatility_compression'] = 1 / (data['true_range'] / data['tr_rolling_median']).replace([np.inf, -np.inf], np.nan)
    
    # Volume Confirmation
    data['volume_rolling_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_signal'] = np.sign(data['volume'] / data['volume_rolling_median'] - 1)
    
    # Interaction and Persistence Logic
    # Breakout-Rejection Interaction
    data['breakout_diff'] = data['high_breakout'] - data['low_breakout']
    data['breakout_rejection_interaction'] = data['breakout_diff'] * data['net_rejection']
    data['breakout_rejection_persistence'] = data['breakout_rejection_interaction'].rolling(window=3, min_periods=2).sum()
    
    # Compression-Enhanced Signal
    data['compression_enhanced'] = data['breakout_rejection_persistence'] * data['volatility_compression'] * data['volume_signal']
    
    # Final Factor Construction
    data['factor_10d_sum'] = data['compression_enhanced'].rolling(window=10, min_periods=7).sum()
    
    # Calculate cross-sectional rank
    factor_values = []
    for date in data.index:
        date_data = data.loc[:date].tail(15)  # Use only past 15 days for ranking
        if len(date_data) >= 10:  # Minimum periods for meaningful ranking
            current_value = date_data['factor_10d_sum'].iloc[-1]
            rank_value = (date_data['factor_10d_sum'].rank(pct=True).iloc[-1] - 0.5) * 2  # Normalize to [-1, 1]
            factor_values.append(rank_value)
        else:
            factor_values.append(np.nan)
    
    factor_series = pd.Series(factor_values, index=data.index, name='factor')
    return factor_series
