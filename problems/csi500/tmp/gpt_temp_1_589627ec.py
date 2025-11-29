import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range for volatility breakout
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 10-day Average True Range for volatility baseline
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    data['vol_threshold'] = data['atr_10'] * 1.5
    
    # Generate volatility breakout signal
    breakout_signal = np.zeros(len(data))
    breakout_signal[data['true_range'] > data['vol_threshold']] = 1
    breakout_signal[data['true_range'] < (data['vol_threshold'] * 0.7)] = -1
    
    # Calculate price momentum components
    data['close_prev'] = data['close'].shift(1)
    data['close_3d_ago'] = data['close'].shift(3)
    data['price_return_3d'] = (data['close'] / data['close_3d_ago']) - 1
    data['price_return_1d'] = (data['close'] / data['close_prev']) - 1
    
    # Calculate volume momentum components
    data['volume_prev'] = data['volume'].shift(1)
    data['volume_3d_ago'] = data['volume'].shift(3)
    data['volume_return_3d'] = (data['volume'] / data['volume_3d_ago']) - 1
    data['volume_return_1d'] = (data['volume'] / data['volume_prev']) - 1
    
    # Calculate momentum differences
    data['price_momentum_diff'] = data['price_return_3d'] - data['price_return_1d']
    data['volume_momentum_diff'] = data['volume_return_3d'] - data['volume_return_1d']
    
    # Generate divergence signal
    divergence_signal = np.zeros(len(data))
    # Positive: price momentum decreasing but volume momentum increasing
    divergence_signal[(data['price_momentum_diff'] < 0) & (data['volume_momentum_diff'] > 0)] = 1
    # Negative: price momentum increasing but volume momentum decreasing
    divergence_signal[(data['price_momentum_diff'] > 0) & (data['volume_momentum_diff'] < 0)] = -1
    
    # Calculate recent price trend for contextual weighting
    data['close_5d_slope'] = data['close'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Calculate trend strength for contextual weighting
    trend_strength = abs(data['close_5d_slope']) / data['close'].rolling(window=5, min_periods=1).std()
    trend_strength = trend_strength.fillna(0)
    
    # Apply contextual weighting
    # Strong trend favors breakout signal, weak trend favors divergence signal
    breakout_weight = trend_strength
    divergence_weight = 1 - trend_strength
    
    # Combine weighted signals
    weighted_breakout = breakout_signal * breakout_weight
    weighted_divergence = divergence_signal * divergence_weight
    
    # Generate final alpha factor
    alpha_factor = weighted_breakout + weighted_divergence
    
    return pd.Series(alpha_factor, index=data.index)
