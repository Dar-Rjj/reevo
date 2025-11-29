import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Path Fractal Analysis
    data['volatility_5d'] = data['high'].rolling(window=5).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=False)
    data['volatility_10d'] = data['high'].rolling(window=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=False)
    
    data['short_term_complexity'] = np.log(data['volatility_5d']) / np.log(5)
    data['medium_term_complexity'] = np.log(data['volatility_10d']) / np.log(10)
    data['volatility_complexity_ratio'] = data['short_term_complexity'] / data['medium_term_complexity']
    
    # Session-Based Volatility-Volume Dynamics
    # Using first 30min and last 30min approximations (assuming 6.5 hour trading day)
    data['volume_total'] = data['volume'].rolling(window=1).sum()  # Daily volume
    data['volume_first_30min'] = data['volume'] * 0.0769  # Approximation: 30min/390min
    data['volume_last_30min'] = data['volume'] * 0.0769   # Approximation: 30min/390min
    
    data['morning_volatility_efficiency'] = ((data['high'] - data['open']) / 
                                           (data['high'] - data['low'])) * \
                                          (data['volume_first_30min'] / data['volume_total'])
    
    data['afternoon_volatility_reversal'] = ((data['close'] - data['low']) / 
                                            (data['high'] - data['low'])) * \
                                           np.sign(data['open'] - data['close'].shift(1))
    
    data['volume_first_hour'] = data['volume'] * 0.1538  # Approximation: 60min/390min
    data['volume_last_hour'] = data['volume'] * 0.1538   # Approximation: 60min/390min
    data['volume_timing_volatility_skew'] = (data['volume_first_hour'] / data['volume_total']) - \
                                           (data['volume_last_hour'] / data['volume_total'])
    
    # Range Compression-Expansion Dynamics
    data['range_compression_ratio'] = (data['high'] - data['low']) / \
                                     (data['high'].shift(5) - data['low'].shift(5))
    
    data['intraday_fracture'] = np.abs((data['high'] + data['low']) / 2 - data['open']) / \
                               (data['high'] - data['low'])
    
    data['range_completion_efficiency'] = (data['close'] - data['low']) / \
                                        (data['high'] - data['low'])
    
    # Breakout Path Complexity
    data['breakout_strength'] = ((data['close'] > data['high'].shift(1)) | 
                                (data['close'] < data['low'].shift(1))) * \
                               (data['high'] - data['low'])
    
    data['liquidity_confirmation'] = (data['amount'] / (data['high'] - data['low'])) * \
                                    data['breakout_strength']
    
    data['breakout_volatility_5d'] = data['high'].rolling(window=5).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=False)
    data['breakout_path_complexity'] = np.log(data['breakout_strength'] * data['breakout_volatility_5d']) / np.log(5)
    
    # Volatility-Volume Timing Integration
    data['morning_concentration_volatility'] = (data['volume_first_30min'] / data['volume_total']) * \
                                              (data['high'] - data['low'])
    
    data['afternoon_concentration_volatility'] = (data['volume_last_30min'] / data['volume_total']) * \
                                               (data['high'] - data['low'])
    
    data['volume_timing_divergence'] = data['morning_concentration_volatility'] - \
                                     data['afternoon_concentration_volatility']
    
    # Composite Alpha Construction
    data['complexity_timing_divergence'] = data['volatility_complexity_ratio'] * \
                                         data['volume_timing_volatility_skew']
    
    data['breakout_efficiency_momentum'] = data['breakout_strength'] * \
                                         data['range_completion_efficiency']
    
    data['path_compression_reversal'] = data['volatility_complexity_ratio'] * \
                                      data['range_compression_ratio'] * \
                                      data['volume_timing_divergence']
    
    # Final alpha factor - weighted combination
    alpha = (0.4 * data['complexity_timing_divergence'] + 
             0.3 * data['breakout_efficiency_momentum'] + 
             0.3 * data['path_compression_reversal'])
    
    return alpha
