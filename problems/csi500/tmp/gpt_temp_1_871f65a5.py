import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # Short-term volatility (3-day rolling std)
    data['short_term_vol'] = data['daily_range'].rolling(window=3, min_periods=2).std()
    
    # Medium-term volatility (10-day rolling std)
    data['medium_term_vol'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    
    # Volatility Regime Ratio
    data['vol_regime_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_regime_ratio'] = data['vol_regime_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility-Clustered Range Signals
    data['prev_range'] = (data['high'] - data['low']).shift(1)
    data['range_expansion'] = (data['high'] - data['low']) / data['prev_range']
    data['range_expansion'] = data['range_expansion'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility Persistence
    data['close_5d_std'] = data['close'].rolling(window=5, min_periods=3).std()
    data['close_10d_std'] = data['close'].rolling(window=10, min_periods=5).std()
    data['vol_persistence'] = data['close_5d_std'] / data['close_10d_std']
    data['vol_persistence'] = data['vol_persistence'].replace([np.inf, -np.inf], np.nan)
    
    # Regime-Adaptive Range
    data['regime_adaptive_range'] = data['range_expansion'] * data['vol_persistence'] * data['vol_regime_ratio']
    
    # Price Reversal Component
    data['prev_close_to_open'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['open_to_close'] = (data['close'] - data['open']) / data['open']
    
    # Reversal Indicator
    data['reversal_indicator'] = -np.sign(data['prev_close_to_open']) * data['open_to_close']
    
    # True Range Calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Core Reversal Signal
    data['core_reversal'] = data['reversal_indicator'] / data['true_range']
    data['core_reversal'] = data['core_reversal'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Efficiency Signals
    data['price_change'] = data['close'] - data['open']
    data['volume_20d_mean'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_deviation'] = data['volume'] / data['volume_20d_mean']
    
    # High volume periods (above 20-day mean)
    high_volume_mask = data['volume'] > data['volume_20d_mean']
    data['volume_clustering_efficiency'] = np.where(
        high_volume_mask,
        data['price_change'] / data['volume'],
        0
    )
    
    # Amount Efficiency
    data['amount_efficiency'] = data['amount'] / (data['high'] - data['low'])
    data['amount_efficiency'] = data['amount_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Divergence
    data['return_sign'] = np.sign(data['price_change'])
    data['volume_dev_sign'] = np.sign(data['volume_deviation'] - 1)
    data['divergence_score'] = (data['return_sign'] != data['volume_dev_sign']).astype(float)
    
    # Volume Timing Efficiency
    data['lag_return'] = data['price_change'].shift(1)
    
    # Calculate rolling correlations
    vol_return_corr = []
    vol_lag_return_corr = []
    
    for i in range(len(data)):
        if i >= 10:  # Minimum period for correlation
            vol_window = data['volume'].iloc[i-9:i+1]
            return_window = data['price_change'].iloc[i-9:i+1]
            lag_return_window = data['lag_return'].iloc[i-9:i+1]
            
            if len(vol_window) >= 5 and len(return_window) >= 5:
                vol_return_corr.append(vol_window.corr(return_window))
                vol_lag_return_corr.append(vol_window.corr(lag_return_window))
            else:
                vol_return_corr.append(np.nan)
                vol_lag_return_corr.append(np.nan)
        else:
            vol_return_corr.append(np.nan)
            vol_lag_return_corr.append(np.nan)
    
    data['vol_return_corr'] = vol_return_corr
    data['vol_lag_return_corr'] = vol_lag_return_corr
    data['volume_timing_efficiency'] = data['vol_return_corr'] - data['vol_lag_return_corr']
    
    # Adaptive Alpha Construction
    data['volume_weighted_component'] = data['core_reversal'] * data['volume_clustering_efficiency'] * data['amount_efficiency']
    data['regime_adaptive_multiplier'] = data['volume_weighted_component'] * data['regime_adaptive_range']
    data['final_alpha'] = data['regime_adaptive_multiplier'] * data['divergence_score'] * data['volume_timing_efficiency']
    
    # Return the final alpha series
    return data['final_alpha']
