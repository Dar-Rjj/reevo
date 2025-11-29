import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Reversal with Volume Exhaustion Factor
    Combines price reversal signals with volume exhaustion patterns
    """
    data = df.copy()
    
    # Price Reversal Components
    # Short-Term Overreaction Detection
    data['upward_exhaustion'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['downward_exhaustion'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Previous day gap analysis
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fill'] = np.where(
        (data['gap'] > 0) & (data['close'] < data['open']), 
        (data['open'] - data['close']) / data['gap'].abs(), 
        np.where(
            (data['gap'] < 0) & (data['close'] > data['open']), 
            (data['close'] - data['open']) / data['gap'].abs(), 
            0
        )
    )
    
    # Medium-Term Mean Reversion
    data['price_median_8d'] = data['close'].rolling(window=8, min_periods=5).median()
    data['price_deviation'] = (data['close'] - data['price_median_8d']) / data['price_median_8d']
    
    # Reversal Timing - Failed breakouts
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['failed_breakout_up'] = np.where(
        (data['high'] == data['high_5d']) & (data['close'] < data['open']),
        (data['high'] - data['close']) / data['close'],
        0
    )
    data['failed_breakout_down'] = np.where(
        (data['low'] == data['low_5d']) & (data['close'] > data['open']),
        (data['close'] - data['low']) / data['close'],
        0
    )
    
    # Volume Exhaustion Signals
    # Volume Climax Detection
    data['volume_percentile_15d'] = data['volume'].rolling(window=15, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.8)) if len(x.dropna()) >= 10 else 0
    )
    data['volume_spike_rejection'] = np.where(
        (data['volume_percentile_15d'] == 1) & 
        ((data['upward_exhaustion'] > 0.7) | (data['downward_exhaustion'] > 0.7)),
        1, 0
    )
    
    # Volume-Price Divergence
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x.dropna()) >= 3 else 0
    )
    data['price_slope_5d'] = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x.dropna()) >= 3 else 0
    )
    data['volume_price_divergence'] = np.where(
        (data['volume_slope_5d'] * data['price_slope_5d'] < 0),
        np.abs(data['volume_slope_5d']) / (np.abs(data['price_slope_5d']) + 1e-8),
        0
    )
    
    # Volume Dry-up Signals
    data['volume_ratio_3d'] = data['volume'] / data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_contraction'] = np.where(
        data['volume_ratio_3d'] < 0.7,
        1 - data['volume_ratio_3d'],
        0
    )
    
    # Factor Integration
    # Combine Reversal and Exhaustion Signals
    reversal_strength = (
        data['upward_exhaustion'] * data['failed_breakout_up'] - 
        data['downward_exhaustion'] * data['failed_breakout_down'] +
        data['price_deviation'] * -1 +  # Mean reversion component
        data['gap_fill'] * np.sign(data['gap']) * -1
    )
    
    volume_exhaustion = (
        data['volume_spike_rejection'] +
        data['volume_price_divergence'] +
        data['volume_contraction']
    )
    
    # Confirmation Filters
    # Price level validation
    data['near_high_10d'] = (data['close'] / data['high'].rolling(window=10, min_periods=7).max() > 0.95).astype(int)
    data['near_low_10d'] = (data['close'] / data['low'].rolling(window=10, min_periods=7).min() < 1.05).astype(int)
    
    price_validation = np.where(
        data['near_high_10d'] == 1, -1,
        np.where(data['near_low_10d'] == 1, 1, 0)
    )
    
    # Time-based weighting
    time_weight = 1.0  # Equal weighting for simplicity
    
    # Signal Generation
    raw_factor = reversal_strength * volume_exhaustion * price_validation * time_weight
    
    # Final factor with smoothing
    factor = raw_factor.rolling(window=3, min_periods=2).mean()
    
    return factor
