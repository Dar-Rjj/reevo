import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum with Volume Confirmation
    # Calculate Intraday Price Momentum
    mid_price = (data['open'] + data['close']) / 2
    data['normalized_price_range'] = (data['high'] - data['low']) / mid_price
    data['price_acceleration'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['mid_point_reversion'] = np.abs((data['high'] + data['low']) / 2 - mid_price) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Compute Volume Divergence Signals
    price_change = np.abs(data['close'] - data['open']).replace(0, np.nan)
    data['volume_to_price_ratio'] = data['volume'] / price_change
    
    # 5-day rolling median volume
    data['rolling_median_volume'] = data['volume'].rolling(window=5, min_periods=1).median()
    data['abnormal_volume_detection'] = data['volume'] / data['rolling_median_volume']
    
    # Volume persistence (using shift(1) for previous day)
    data['previous_volume'] = data['volume'].shift(1)
    data['volume_persistence'] = data['volume'] / data['previous_volume']
    
    # Create Price-Volume Divergence Factors
    data['momentum_divergence'] = data['price_acceleration'] * data['volume_to_price_ratio']
    data['range_expansion_factor'] = data['normalized_price_range'] * data['abnormal_volume_detection']
    data['reversion_strength'] = data['mid_point_reversion'] * data['volume_persistence']
    
    # Order Flow Imbalance Estimation
    # Calculate Imbalance Proxies
    data['tick_imbalance_proxy'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_weighted_price_skew'] = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan) * data['volume']
    data['effective_tick_size'] = (data['high'] - data['low']) / data['volume'].replace(0, np.nan)
    
    # Compute Flow Persistence
    data['previous_close'] = data['close'].shift(1)
    data['previous_open'] = data['open'].shift(1)
    data['flow_direction_consistency'] = np.sign(data['close'] - data['open']) * np.sign(data['previous_close'] - data['previous_open'])
    data['flow_magnitude_ratio'] = np.abs(data['close'] - data['open']) / np.abs(data['previous_close'] - data['previous_open']).replace(0, np.nan)
    data['intraday_flow_acceleration'] = data['tick_imbalance_proxy'] * data['volume']
    
    # Generate Order Flow Factors
    data['imbalance_momentum'] = data['tick_imbalance_proxy'] * data['flow_direction_consistency']
    data['volume_weighted_flow'] = data['volume_weighted_price_skew'] * data['flow_magnitude_ratio']
    data['flow_efficiency'] = data['intraday_flow_acceleration'] / data['effective_tick_size']
    
    # Construct Composite Alpha Factor
    # Combine Divergence and Flow Components
    primary_factor = data['momentum_divergence'] * data['imbalance_momentum']
    secondary_factor = data['range_expansion_factor'] * data['volume_weighted_flow']
    tertiary_factor = data['reversion_strength'] * data['flow_efficiency']
    
    # Apply Dynamic Weighting Scheme
    # Calculate 10-day rolling volatility
    primary_vol = primary_factor.rolling(window=10, min_periods=1).std()
    secondary_vol = secondary_factor.rolling(window=10, min_periods=1).std()
    tertiary_vol = tertiary_factor.rolling(window=10, min_periods=1).std()
    
    # Inverse volatility weighting
    vol_sum = primary_vol + secondary_vol + tertiary_vol
    primary_weight = (secondary_vol + tertiary_vol) / (2 * vol_sum).replace(0, np.nan)
    secondary_weight = (primary_vol + tertiary_vol) / (2 * vol_sum).replace(0, np.nan)
    tertiary_weight = (primary_vol + secondary_vol) / (2 * vol_sum).replace(0, np.nan)
    
    # Incorporate component persistence via 5-day autocorrelation
    primary_persistence = primary_factor.rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 1, raw=False)
    secondary_persistence = secondary_factor.rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 1, raw=False)
    tertiary_persistence = tertiary_factor.rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 1, raw=False)
    
    # Adjust weights by persistence
    primary_weight_adj = primary_weight * (1 + primary_persistence.fillna(0))
    secondary_weight_adj = secondary_weight * (1 + secondary_persistence.fillna(0))
    tertiary_weight_adj = tertiary_weight * (1 + tertiary_persistence.fillna(0))
    
    # Normalize weights
    weight_sum = primary_weight_adj + secondary_weight_adj + tertiary_weight_adj
    primary_weight_final = primary_weight_adj / weight_sum.replace(0, np.nan)
    secondary_weight_final = secondary_weight_adj / weight_sum.replace(0, np.nan)
    tertiary_weight_final = tertiary_weight_adj / weight_sum.replace(0, np.nan)
    
    # Volatility-weighted combination
    composite_factor = (primary_factor * primary_weight_final + 
                       secondary_factor * secondary_weight_final + 
                       tertiary_factor * tertiary_weight_final)
    
    # Incorporate Session-Based Adjustment (simplified proxy)
    # Early Session Dominance: using first hour range as proxy for early session activity
    data['early_session_range'] = (data['high'].rolling(window=5, min_periods=1).max() - 
                                  data['low'].rolling(window=5, min_periods=1).min())
    data['daily_range'] = data['high'] - data['low']
    early_session_dominance = data['early_session_range'] / data['daily_range'].replace(0, np.nan)
    
    # Final Factor
    final_factor = composite_factor * early_session_dominance
    
    return final_factor
