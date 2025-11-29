import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Price Position Ratio
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Compute Volatility-Adjusted Price Position
    data['vol_adj_price_position'] = data['price_position'] / (data['true_range'] + 1e-8)
    
    # Derive Reversal Signal
    data['vol_adj_reversal'] = 1 - data['vol_adj_price_position']
    
    # Apply magnitude weighting using absolute price change
    data['abs_price_change'] = abs(data['close'] - data['prev_close'])
    data['weighted_reversal'] = data['vol_adj_reversal'] * data['abs_price_change']
    
    # Calculate Volume Momentum Component
    data['volume_roc'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    
    # Calculate Volume-Price Consistency (5-day correlation)
    data['abs_return'] = abs(data['close'] / data['prev_close'] - 1)
    
    # Rolling correlation between volume and absolute returns
    vol_price_corr = []
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-4:i+1]
            if len(window_data) >= 2:
                corr = window_data['volume'].corr(window_data['abs_return'])
                vol_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                vol_price_corr.append(0)
        else:
            vol_price_corr.append(0)
    
    data['vol_price_corr'] = vol_price_corr
    
    # Enhanced volume momentum signal
    data['enhanced_volume_momentum'] = data['volume_roc'] * data['vol_price_corr']
    
    # Multiply Volatility-Adjusted Reversal by Volume Momentum
    data['raw_factor'] = data['weighted_reversal'] * data['enhanced_volume_momentum']
    
    # Apply pattern persistence weighting
    data['reversal_direction'] = np.sign(data['raw_factor'])
    data['pattern_streak'] = 0
    
    for i in range(1, len(data)):
        if data['reversal_direction'].iloc[i] == data['reversal_direction'].iloc[i-1]:
            data.loc[data.index[i], 'pattern_streak'] = data['pattern_streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'pattern_streak'] = 1
    
    # Weight by volatility-adjusted magnitude
    data['persistence_weight'] = data['pattern_streak'] * data['abs_price_change'] / (data['true_range'] + 1e-8)
    data['weighted_factor'] = data['raw_factor'] * data['persistence_weight']
    
    # Calculate 3-day cumulative factor for signal strength
    data['final_factor'] = data['weighted_factor'].rolling(window=3, min_periods=1).sum()
    
    return data['final_factor']
