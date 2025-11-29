import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Movement Asymmetry Detection
    data['abs_price_move'] = np.abs(data['close'] - data['open'])
    data['up_move'] = np.maximum(0, data['close'] - data['open'])
    data['down_move'] = np.maximum(0, data['open'] - data['close'])
    
    # Asymmetry Ratio Calculation
    data['up_ratio'] = np.where(data['abs_price_move'] > 0, 
                               data['up_move'] / data['abs_price_move'], 0)
    data['down_ratio'] = np.where(data['abs_price_move'] > 0, 
                                 data['down_move'] / data['abs_price_move'], 0)
    
    # Asymmetry Persistence Tracking
    data['consecutive_up'] = 0
    data['consecutive_down'] = 0
    
    for i in range(1, len(data)):
        if data['up_ratio'].iloc[i] > 0.6:
            data['consecutive_up'].iloc[i] = data['consecutive_up'].iloc[i-1] + 1
        else:
            data['consecutive_up'].iloc[i] = 0
            
        if data['down_ratio'].iloc[i] > 0.6:
            data['consecutive_down'].iloc[i] = data['consecutive_down'].iloc[i-1] + 1
        else:
            data['consecutive_down'].iloc[i] = 0
    
    # Volume Asymmetry Components (using amount as proxy for volume concentration)
    data['daily_range'] = data['high'] - data['low']
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Estimate volume concentration using price-based proxies
    data['volume_variation'] = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=5).mean()
    data['price_volatility'] = data['daily_range'].rolling(window=5).std() / data['mid_price'].rolling(window=5).mean()
    
    # Volume timing asymmetry proxy
    data['open_close_volume_ratio'] = np.where(data['volume'] > 0, 
                                              (data['close'] - data['open']) / data['volume'], 0)
    
    # Regime Classification System
    data['price_range_median'] = data['daily_range'].rolling(window=20, min_periods=10).median()
    data['volume_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    
    data['range_state'] = np.where(data['daily_range'] > data['price_range_median'], 1, -1)
    data['volume_state'] = np.where(data['volume'] > data['volume_median'], 1, -1)
    
    # Combined regime mapping
    data['regime'] = data['range_state'] * 2 + data['volume_state']
    
    # Asymmetry Divergence Components
    data['price_volume_divergence'] = data['up_ratio'] * data['volume_variation']
    data['timing_mismatch'] = data['open_close_volume_ratio'] * data['price_volatility']
    
    # Persistence Divergence
    data['persistence_divergence'] = (data['consecutive_up'] - data['consecutive_down']) * data['volume_variation']
    
    # Alpha Construction - Regime-Specific Asymmetry Scoring
    regime_factors = {}
    
    # High range regimes (range_state = 1)
    high_range_mask = data['range_state'] == 1
    regime_factors['high_range'] = np.where(high_range_mask, 
                                          data['timing_mismatch'] * 0.6 + data['price_volume_divergence'] * 0.4, 0)
    
    # Low range regimes (range_state = -1)
    low_range_mask = data['range_state'] == -1
    regime_factors['low_range'] = np.where(low_range_mask, 
                                         data['persistence_divergence'] * 0.7 + data['up_ratio'] * 0.3, 0)
    
    # High volume regimes (volume_state = 1)
    high_volume_mask = data['volume_state'] == 1
    regime_factors['high_volume'] = np.where(high_volume_mask, 
                                           data['volume_variation'] * 0.5 + data['price_volume_divergence'] * 0.5, 0)
    
    # Low volume regimes (volume_state = -1)
    low_volume_mask = data['volume_state'] == -1
    regime_factors['low_volume'] = np.where(low_volume_mask, 
                                          data['up_ratio'] * 0.8 + data['down_ratio'] * -0.2, 0)
    
    # Combine regime-specific factors
    data['regime_factor'] = (regime_factors['high_range'] + regime_factors['low_range'] + 
                           regime_factors['high_volume'] + regime_factors['low_volume'])
    
    # Apply persistence filters
    data['persistence_filter'] = np.where((data['consecutive_up'] > 2) | (data['consecutive_down'] > 2), 1.2, 1.0)
    data['persistence_filter'] = np.where((data['consecutive_up'] > 5) | (data['consecutive_down'] > 5), 1.5, data['persistence_filter'])
    
    # Final Alpha Output
    data['alpha'] = data['regime_factor'] * data['persistence_filter']
    
    # Normalize and clean
    alpha_series = data['alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
