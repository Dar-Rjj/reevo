import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition with Price-Volume Asymmetry Analysis
    """
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = data['close'] - data['open']
    data['abs_price_change'] = abs(data['price_change'])
    
    # Volatility regime detection
    data['range_ratio'] = data['daily_range'] / data['daily_range'].shift(1)
    data['volatility_persistence'] = (data['daily_range'].shift(5) / 
                                    data['daily_range'].shift(10)).replace([np.inf, -np.inf], np.nan)
    data['expansion_acceleration'] = (data['daily_range'].shift(2) / 
                                    data['daily_range'].shift(5)).replace([np.inf, -np.inf], np.nan)
    
    # Identify compression periods (consecutive low volatility)
    compression_threshold = 0.7
    data['is_compressed'] = data['range_ratio'] < compression_threshold
    data['compression_cluster'] = data['is_compressed'].astype(int).rolling(window=5, min_periods=1).sum()
    
    # Identify expansion periods (consecutive high volatility)
    expansion_threshold = 1.3
    data['is_expanded'] = data['range_ratio'] > expansion_threshold
    data['expansion_cluster'] = data['is_expanded'].astype(int).rolling(window=5, min_periods=1).sum()
    
    # Regime transition detection
    data['compression_to_expansion'] = ((data['is_compressed'].shift(1) == True) & 
                                      (data['is_expanded'] == True)).astype(int)
    data['expansion_to_compression'] = ((data['is_expanded'].shift(1) == True) & 
                                      (data['is_compressed'] == True)).astype(int)
    
    # Price-volume asymmetry analysis
    # Upward movement asymmetry
    up_mask = data['close'] > data['open']
    data['up_volume_asymmetry'] = np.where(up_mask, 
                                          data['price_change'] / data['volume'], 
                                          0)
    data['up_volume_flat_asymmetry'] = np.where(up_mask,
                                               data['price_change'] / (data['volume'] / data['volume'].shift(1)),
                                               0)
    
    # Downward movement asymmetry
    down_mask = data['close'] < data['open']
    data['down_volume_asymmetry'] = np.where(down_mask,
                                            -data['price_change'] / data['volume'],
                                            0)
    data['down_volume_flat_asymmetry'] = np.where(down_mask,
                                                 -data['price_change'] / (data['volume'] / data['volume'].shift(1)),
                                                 0)
    
    # Strong divergence detection
    data['strong_up_weak_volume'] = ((data['price_change'] > data['price_change'].rolling(5).mean()) & 
                                   (data['volume'] < data['volume'].rolling(5).mean()) & up_mask).astype(int)
    data['strong_down_weak_volume'] = ((data['price_change'] < -data['abs_price_change'].rolling(5).mean()) & 
                                     (data['volume'] < data['volume'].rolling(5).mean()) & down_mask).astype(int)
    
    # Multi-timeframe asymmetry validation
    # Short-term (1-3 days)
    data['short_term_asymmetry'] = (data['up_volume_asymmetry'] + data['down_volume_asymmetry']).rolling(3).mean()
    data['short_term_persistence'] = data['short_term_asymmetry'].rolling(3).std()
    
    # Medium-term (5-10 days)
    data['medium_term_asymmetry'] = (data['up_volume_asymmetry'] + data['down_volume_asymmetry']).rolling(10).mean()
    data['medium_term_trend'] = data['medium_term_asymmetry'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Long-term (20-40 days)
    data['long_term_asymmetry'] = (data['up_volume_asymmetry'] + data['down_volume_asymmetry']).rolling(40).mean()
    data['long_term_breakpoint'] = (data['long_term_asymmetry'] > 
                                  data['long_term_asymmetry'].rolling(20).mean() + 
                                  data['long_term_asymmetry'].rolling(20).std()).astype(int)
    
    # Regime-asymmetry interaction analysis
    # Compression regime behavior
    compression_mask = data['compression_cluster'] >= 3
    data['compression_asymmetry'] = np.where(compression_mask,
                                           data['short_term_asymmetry'],
                                           0)
    data['compression_asymmetry_persistence'] = np.where(compression_mask,
                                                       data['short_term_persistence'],
                                                       0)
    
    # Expansion regime behavior
    expansion_mask = data['expansion_cluster'] >= 3
    data['expansion_asymmetry'] = np.where(expansion_mask,
                                         data['short_term_asymmetry'],
                                         0)
    
    # Transition phase dynamics
    transition_mask = (data['compression_to_expansion'] == 1) | (data['expansion_to_compression'] == 1)
    data['transition_asymmetry'] = np.where(transition_mask,
                                          data['short_term_asymmetry'] * 1.5,  # Amplification during transitions
                                          0)
    
    # Compute final factor
    # Base regime-asymmetry score
    data['regime_asymmetry_base'] = (
        data['compression_asymmetry'] * 0.4 +
        data['expansion_asymmetry'] * 0.3 +
        data['transition_asymmetry'] * 0.3
    )
    
    # Multi-timeframe validation weights
    data['timeframe_validation'] = (
        data['short_term_persistence'] * 0.4 +
        abs(data['medium_term_trend']) * 0.3 +
        data['long_term_breakpoint'] * 0.3
    )
    
    # Final factor with regime transition timing and validation
    data['factor'] = (
        data['regime_asymmetry_base'] * 
        (1 + data['timeframe_validation']) *
        (1 + 0.2 * (data['compression_to_expansion'] + data['expansion_to_compression']))
    )
    
    # Risk adjustment through regime stability
    regime_stability = 1 / (1 + data['compression_to_expansion'].rolling(10).sum() + 
                          data['expansion_to_compression'].rolling(10).sum())
    data['factor'] = data['factor'] * regime_stability
    
    # Clean and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
