import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns and basic metrics
    data['returns'] = data['close'].pct_change()
    
    # 1. Intraday Volatility Surge
    # Current day's high-low range
    data['intraday_range'] = data['high'] - data['low']
    
    # Historical volatility baseline (20-day rolling std of close-to-close returns)
    data['hist_vol_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Volatility ratio (avoid division by zero)
    data['vol_ratio'] = data['intraday_range'] / (data['hist_vol_20d'] + 1e-8)
    
    # 2. Price Acceleration Pattern
    # Calculate short-term momentum (5-period returns)
    data['momentum_5'] = data['close'].pct_change(periods=5)
    
    # Calculate acceleration as change in momentum slope
    data['momentum_change'] = data['momentum_5'] - data['momentum_5'].shift(5)
    
    # Breakout confirmation flag
    data['breakout_flag'] = ((data['vol_ratio'] > 1.5) & 
                            (data['momentum_change'] > 0)).astype(int)
    
    # Acceleration weight (normalized momentum change)
    momentum_range = data['momentum_change'].rolling(window=50, min_periods=20).apply(
        lambda x: x.max() - x.min(), raw=True)
    data['accel_weight'] = data['momentum_change'] / (momentum_range + 1e-8)
    
    # 3. Liquidity Conditions
    # Volume-weighted price efficiency
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    data['price_efficiency'] = 1 - (abs(data['close'] - data['vwap']) / data['close'])
    
    # Intraday volume concentration (30-period rolling windows)
    data['volume_30m_avg'] = data['volume'].rolling(window=30, min_periods=15).mean()
    data['volume_30m_max'] = data['volume'].rolling(window=30, min_periods=15).max()
    data['volume_concentration'] = data['volume_30m_max'] / (data['volume_30m_avg'] + 1e-8)
    
    # Liquidity quality score
    data['liquidity_score'] = data['price_efficiency'] / (data['volume_concentration'] + 1e-8)
    
    # Normalize liquidity score
    liquidity_range = data['liquidity_score'].rolling(window=50, min_periods=20).apply(
        lambda x: x.max() - x.min(), raw=True)
    data['liquidity_quality'] = data['liquidity_score'] / (liquidity_range + 1e-8)
    
    # 4. Combine Components with Time Decay
    # Raw signal component
    data['raw_signal'] = (data['vol_ratio'] * data['breakout_flag'] * 
                         data['accel_weight'] * data['liquidity_quality'])
    
    # Apply exponential time decay (decay factor = 0.95 per period)
    decay_weights = []
    decay_factor = 0.95
    for i in range(len(data)):
        if i == 0:
            decay_weights.append(1.0)
        else:
            decay_weights.append(decay_weights[-1] * decay_factor)
    
    # Reverse weights so recent periods have higher weight
    decay_weights = decay_weights[::-1]
    
    # Apply decay using expanding window with decay weights
    final_factor = []
    for i in range(len(data)):
        if i < 20:  # Minimum periods for meaningful calculation
            final_factor.append(np.nan)
        else:
            window_data = data['raw_signal'].iloc[:i+1]
            window_weights = decay_weights[:len(window_data)]
            # Normalize weights
            window_weights = np.array(window_weights[-len(window_data):])
            window_weights = window_weights / window_weights.sum()
            weighted_signal = (window_data * window_weights).sum()
            final_factor.append(weighted_signal)
    
    # Create output series
    factor_series = pd.Series(final_factor, index=data.index)
    
    return factor_series
