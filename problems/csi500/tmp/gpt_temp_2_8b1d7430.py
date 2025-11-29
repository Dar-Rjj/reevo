import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['range'] = data['high'] - data['low']
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['open_to_close'] = data['close'] - data['open']
    
    # 1. Intraday Momentum Fragmentation Patterns
    # Morning vs Afternoon momentum
    data['morning_momentum'] = data['high'] - data['open']
    data['afternoon_momentum'] = data['close'] - data['mid_price']
    data['momentum_divergence'] = np.abs(data['morning_momentum'] - data['afternoon_momentum'])
    
    # Momentum direction consistency
    data['momentum_consistency'] = np.where(
        (data['morning_momentum'] * data['afternoon_momentum']) > 0, 1, -1
    )
    
    # Price path efficiency
    data['intraday_abs_change'] = np.abs(data['high'] - data['open']) + np.abs(data['low'] - data['open']) + np.abs(data['close'] - data['mid_price'])
    data['price_efficiency'] = np.abs(data['open_to_close']) / (data['intraday_abs_change'] + 1e-8)
    
    # Gap fill momentum
    data['gap'] = data['open'] - data['close'].shift(1)
    data['gap_fill_ratio'] = np.where(
        data['gap'] != 0,
        (data['close'] - data['open']) / (data['gap'] + 1e-8),
        0
    )
    
    # 2. Failed momentum clustering
    # Rolling extremes comparison
    data['roll_high_5'] = data['high'].rolling(window=5, min_periods=3).max()
    data['roll_low_5'] = data['low'].rolling(window=5, min_periods=3).min()
    data['extreme_proximity'] = (
        (data['high'] - data['roll_high_5']) / data['roll_high_5'] + 
        (data['low'] - data['roll_low_5']) / data['roll_low_5']
    )
    
    # Momentum exhaustion
    data['morning_velocity'] = data['mid_price'] - data['open']
    data['afternoon_velocity'] = data['close'] - data['mid_price']
    data['velocity_deceleration'] = np.abs(data['afternoon_velocity']) - np.abs(data['morning_velocity'])
    
    # Volume-momentum divergence
    data['volume_price_divergence'] = (
        data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean() - 
        np.abs(data['open_to_close']) / np.abs(data['open_to_close']).rolling(window=5, min_periods=3).mean()
    )
    
    # 3. Volume-Volatility Mismatch
    # Volatility efficiency
    data['range_utilization'] = data['range'] / (data['range'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Volume-volatility correlation (3-day rolling)
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    data['vol_vol_corr'] = rolling_corr(data['volume'], data['range'], 3)
    
    # Volume distribution asymmetry
    data['volume_efficiency'] = data['amount'] / (data['volume'] + 1e-8)
    data['microstructure_noise'] = data['range'] / (data['amount'] + 1e-8)
    
    # 4. Multi-Timeframe Fragmentation
    # Session boundary effects
    data['overnight_momentum'] = data['open'] - data['close'].shift(1)
    data['intraday_momentum'] = data['close'] - data['open']
    
    # Volatility clustering
    data['range_autocorr'] = data['range'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # Liquidity fragmentation
    data['volume_skew'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # 5. Composite Signal Construction
    # Momentum fragmentation component
    momentum_fragmentation = (
        data['momentum_divergence'] * (1 - data['momentum_consistency']) * 
        (1 - data['price_efficiency']) * np.abs(data['gap_fill_ratio'])
    )
    
    # Volume-volatility mismatch component
    vol_vol_mismatch = (
        (1 - data['range_utilization']) * (1 - np.abs(data['vol_vol_corr'])) * 
        data['volume_price_divergence'] * data['microstructure_noise']
    )
    
    # Multi-timeframe adjustment
    timeframe_adjustment = (
        np.abs(data['overnight_momentum']) * data['range_autocorr'] * data['volume_skew']
    )
    
    # Final composite factor
    factor = (
        momentum_fragmentation * vol_vol_mismatch * timeframe_adjustment * 
        data['velocity_deceleration'] * data['extreme_proximity']
    )
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(method='ffill').fillna(0)
    
    return factor
