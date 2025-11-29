import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Compression Fracture Dynamics factor
    Detects compression patterns and evaluates fracture quality for momentum prediction
    """
    
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily ranges and gaps
    data['daily_range'] = data['high'] - data['low']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['open_gap'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Opening compression analysis
    data['opening_range'] = (data['high'].rolling(window=30, min_periods=10).apply(
        lambda x: x.iloc[0] if len(x) > 0 else np.nan
    ) - data['low'].rolling(window=30, min_periods=10).apply(
        lambda x: x.iloc[0] if len(x) > 0 else np.nan
    )) / data['close'].rolling(window=30, min_periods=10).mean()
    
    data['gap_compression'] = data['open_gap'] / (data['true_range'].rolling(window=5).mean() + 1e-8)
    
    # Volume compression detection
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_compression'] = data['volume'] / (data['volume_ma_20'] + 1e-8)
    data['volume_valley'] = (data['volume'] < data['volume_ma_5']) & (data['volume'] < data['volume_ma_20'])
    
    # Extended low-volume periods (3-day rolling)
    data['low_volume_streak'] = data['volume_compression'].rolling(window=3).apply(
        lambda x: sum(x < 0.8) if len(x) == 3 else 0
    )
    
    # Price range compression assessment
    data['range_ma_5'] = data['daily_range'].rolling(window=5).mean()
    data['range_ma_20'] = data['daily_range'].rolling(window=20).mean()
    data['range_compression'] = data['daily_range'] / (data['range_ma_20'] + 1e-8)
    
    # Compression duration analysis (using rolling compression streaks)
    data['compression_streak'] = 0
    for i in range(1, len(data)):
        if (data['range_compression'].iloc[i] < 0.7 and 
            data['volume_compression'].iloc[i] < 0.8):
            data.loc[data.index[i], 'compression_streak'] = data['compression_streak'].iloc[i-1] + 1
    
    # Fracture trigger events - volume spike detection
    data['volume_spike'] = (data['volume'] > data['volume_ma_20'] * 1.5).astype(int)
    
    # Range expansion onset
    data['range_expansion'] = (data['daily_range'] > data['range_ma_5'] * 1.3).astype(int)
    
    # Fracture quality assessment
    data['fracture_signal'] = ((data['volume_spike'] == 1) & 
                              (data['range_expansion'] == 1) & 
                              (data['compression_streak'] >= 2)).astype(int)
    
    # Clean break vs false break (using close relative to range)
    data['intraday_strength'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['clean_break'] = ((data['fracture_signal'] == 1) & 
                          (data['intraday_strength'] > 0.6)).astype(int)
    
    # Post-fracture momentum evaluation (using next day open-to-close performance)
    data['next_day_return'] = (data['close'].shift(-1) - data['open'].shift(-1)) / data['open'].shift(-1)
    
    # For historical analysis, use rolling performance of previous fracture signals
    data['fracture_momentum'] = data['fracture_signal'].rolling(window=10).apply(
        lambda x: np.nanmean(data.loc[x.index, 'next_day_return'].values[x == 1]) 
        if sum(x == 1) > 0 else 0
    )
    
    # Multi-day compression patterns
    data['prev_day_compression'] = data['range_compression'].shift(1)
    data['consecutive_compression'] = ((data['range_compression'] < 0.7) & 
                                      (data['prev_day_compression'] < 0.7)).astype(int)
    
    # Generate final compression signal
    # Compression strength component
    compression_strength = (
        (1 - data['range_compression']) * 0.4 +
        (1 - data['volume_compression']) * 0.3 +
        (data['compression_streak'] / 10) * 0.3
    )
    
    # Fracture quality component
    fracture_quality = (
        data['fracture_signal'] * 0.5 +
        data['clean_break'] * 0.3 +
        data['volume_spike'] * 0.2
    )
    
    # Timing and momentum weighting
    timing_efficiency = (
        (data['opening_range'] < 0.01).astype(int) * 0.4 +
        data['fracture_momentum'].fillna(0) * 0.6
    )
    
    # Multi-day pattern incorporation
    multi_day_pattern = (
        data['consecutive_compression'] * 0.6 +
        data['low_volume_streak'] / 3 * 0.4
    )
    
    # Final factor calculation
    factor = (
        compression_strength * 0.35 +
        fracture_quality * 0.30 +
        timing_efficiency * 0.20 +
        multi_day_pattern * 0.15
    )
    
    # Ensure no future data leakage
    factor = factor.shift(1)  # Use yesterday's signal for today's prediction
    
    return factor
