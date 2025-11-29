import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining volatility regime transitions, 
    price-volume coherence, dollar flow momentum, and range efficiency dynamics.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Transitions
    # Volatility Breakout Efficiency
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_5d'] = data['daily_range'].rolling(window=5).std()
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    data['vol_breakout_ratio'] = data['volatility_5d'] / data['volatility_20d']
    
    # Regime Duration Patterns
    data['high_vol_regime'] = (data['vol_breakout_ratio'] > 1.2).astype(int)
    data['regime_duration'] = data['high_vol_regime'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # 2. Price-Volume Coherence
    # Volume-Price Synchronization
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr'] = data['price_change'].rolling(window=10).corr(data['volume_change'])
    
    # Coherence Breakdown Detection
    data['abs_price_change'] = data['price_change'].abs()
    data['coherence_break'] = (data['abs_price_change'] > 0.02) & (data['price_volume_corr'] < 0)
    
    # 3. Dollar Flow Momentum
    # Multi-Period Flow Alignment
    data['dollar_flow'] = data['close'] * data['volume']
    data['flow_5d'] = data['dollar_flow'].rolling(window=5).mean()
    data['flow_20d'] = data['dollar_flow'].rolling(window=20).mean()
    data['flow_alignment'] = np.sign(data['flow_5d'] - data['flow_5d'].shift(5)) * np.sign(data['flow_20d'] - data['flow_20d'].shift(20))
    
    # Flow Acceleration Patterns
    data['flow_accel_5d'] = (data['flow_5d'] - data['flow_5d'].shift(5)) / data['flow_5d'].shift(5)
    data['flow_accel_20d'] = (data['flow_20d'] - data['flow_20d'].shift(20)) / data['flow_20d'].shift(20)
    data['accel_alignment'] = np.sign(data['flow_accel_5d']) * np.sign(data['flow_accel_20d'])
    
    # 4. Range Efficiency Dynamics
    # Daily Range Utilization
    data['open_to_close'] = (data['close'] - data['open']) / data['open']
    data['range_utilization'] = data['open_to_close'].abs() / data['daily_range']
    
    # Range Breakout Quality
    data['prev_close'] = data['close'].shift(1)
    data['gap_size'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['breakout_quality'] = data['range_utilization'] * np.sign(data['gap_size']) * (1 - data['daily_range'])
    
    # Combine factors with appropriate weights
    data['volatility_factor'] = data['vol_breakout_ratio'] * (1 + 0.1 * data['regime_duration'])
    data['coherence_factor'] = data['price_volume_corr'] - 2 * data['coherence_break'].astype(int)
    data['flow_factor'] = data['flow_alignment'] + 0.5 * data['accel_alignment']
    data['range_factor'] = data['range_utilization'] + data['breakout_quality']
    
    # Final alpha factor combination
    data['alpha_factor'] = (
        0.3 * data['volatility_factor'] +
        0.25 * data['coherence_factor'] +
        0.25 * data['flow_factor'] +
        0.2 * data['range_factor']
    )
    
    # Normalize by cross-sectional z-score
    def cross_sectional_zscore(series):
        return (series - series.mean()) / series.std()
    
    alpha_series = data.groupby(data.index)['alpha_factor'].transform(cross_sectional_zscore)
    
    return alpha_series
