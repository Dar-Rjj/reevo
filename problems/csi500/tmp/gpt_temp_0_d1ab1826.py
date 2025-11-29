import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Price-Volume Divergence Factor
    Combines price efficiency signals with volume anomaly detection
    """
    data = df.copy()
    
    # Price Efficiency Component
    # Normalized Intraday Return
    data['intraday_return'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Price Momentum Consistency
    data['intraday_return_rolling_median'] = data['intraday_return'].rolling(window=3, min_periods=1).median()
    data['price_momentum_consistency'] = data['intraday_return'] - data['intraday_return_rolling_median']
    
    # Price rejection signals
    data['upper_shadow_ratio'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_shadow_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_rejection'] = data['upper_shadow_ratio'] - data['lower_shadow_ratio']
    
    # Opening Gap Adjustment
    data['prev_close'] = data['close'].shift(1)
    data['gap_ratio'] = (data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    
    # Combine price efficiency signals
    data['price_efficiency'] = (
        data['intraday_return'].fillna(0) + 
        data['price_momentum_consistency'].fillna(0) + 
        data['price_rejection'].fillna(0) + 
        data['gap_ratio'].fillna(0)
    )
    
    # Volume Anomaly Component
    # Volume Concentration Pattern
    data['volume_weighted_price'] = (
        (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan) * data['volume']
    )
    data['volume_skewness'] = data['volume_weighted_price'] - data['volume_weighted_price'].rolling(window=5, min_periods=1).mean()
    
    # Volume Breakout Detection
    data['volume_80th_percentile'] = data['volume'].rolling(window=10, min_periods=1).quantile(0.8)
    data['volume_spike'] = data['volume'] / data['volume_80th_percentile'].replace(0, np.nan)
    
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_acceleration'] = (data['volume'] / data['prev_volume'].replace(0, np.nan)) - 1
    
    # Volume-Price Divergence
    data['normalized_range'] = (data['high'] - data['low']) / data['close'].replace(0, np.nan)
    data['high_volume_small_range'] = data['volume_spike'] / (data['normalized_range'] + 1e-6)
    data['low_volume_large_range'] = (1 / (data['volume_spike'] + 1e-6)) * data['normalized_range']
    
    # Combine volume anomaly signals
    data['volume_anomaly'] = (
        data['volume_skewness'].fillna(0) + 
        data['volume_spike'].fillna(0) + 
        data['volume_acceleration'].fillna(0) + 
        data['high_volume_small_range'].fillna(0) - 
        data['low_volume_large_range'].fillna(0)
    )
    
    # Signal Integration
    # Combine Price Efficiency × Volume Anomaly
    data['raw_factor'] = data['price_efficiency'] * data['volume_anomaly']
    
    # Dynamic Weighting
    # Volatility context
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['volatility_context'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    
    # Volume regime
    data['volume_trend'] = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Apply dynamic weighting
    volatility_weight = 1 / (data['volatility_context'] + 1e-6)
    volume_weight = 1 + np.tanh(data['volume_trend'])
    data['weighted_factor'] = data['raw_factor'] * volatility_weight * volume_weight
    
    # Multi-Timeframe Confirmation
    # 1-day vs 3-day price momentum alignment
    data['price_momentum_1d'] = data['close'] / data['prev_close'].replace(0, np.nan) - 1
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3).replace(0, np.nan) - 1
    data['price_alignment'] = np.sign(data['price_momentum_1d']) * np.sign(data['price_momentum_3d'])
    
    # 1-day vs 5-day volume pattern consistency
    data['volume_1d'] = data['volume']
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_consistency'] = data['volume_1d'] / data['volume_5d_avg'].replace(0, np.nan)
    
    # Final factor with multi-timeframe confirmation
    data['final_factor'] = (
        data['weighted_factor'] * 
        (1 + data['price_alignment'].fillna(0)) * 
        (1 + np.tanh(data['volume_consistency'].fillna(0) - 1))
    )
    
    # Clean and return
    factor = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
