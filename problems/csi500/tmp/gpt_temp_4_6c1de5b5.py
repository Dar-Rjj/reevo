import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Identify Momentum Extremes
    # Calculate intraday momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Compute momentum acceleration
    data['momentum_acceleration'] = data['intraday_momentum'] - data['intraday_momentum'].shift(1)
    
    # Detect extreme momentum levels using rolling percentile
    data['momentum_percentile'] = data['intraday_momentum'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # 2. Analyze Volume Confirmation Patterns
    # Compute volume intensity
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_intensity'] = data['volume'] / data['volume_20d_avg']
    
    # Calculate volume persistence (consecutive days above average volume)
    def calc_volume_persistence(series):
        persistence = 0
        for i in range(len(series)-1, -1, -1):
            if series.iloc[i] > 1:
                persistence += 1
            else:
                break
        return persistence
    
    data['volume_persistence'] = data['volume_intensity'].rolling(window=10, min_periods=5).apply(
        calc_volume_persistence, raw=False
    )
    
    # Derive volume-momentum alignment
    data['momentum_direction'] = np.sign(data['intraday_momentum'])
    data['volume_momentum_alignment'] = data['volume_intensity'] * data['momentum_direction']
    
    # 3. Assess Price Rejection Signals
    # Calculate shadow ratios
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['shadow_ratio'] = data['upper_shadow'] / (data['lower_shadow'] + 1e-8)
    
    # Track failed breakout attempts
    data['resistance_level'] = data['high'].rolling(window=5, min_periods=3).max().shift(1)
    data['failed_breakout'] = ((data['high'] > data['resistance_level']) & 
                              (data['close'] < data['resistance_level'])).astype(int)
    
    # Measure reversal strength
    data['price_range'] = data['high'] - data['low']
    data['reversal_strength'] = np.where(
        data['intraday_momentum'] > 0,
        (data['high'] - data['close']) / data['price_range'],
        (data['close'] - data['low']) / data['price_range']
    )
    
    # 4. Generate Composite Factor
    # Weight momentum extremes by volume confirmation strength
    momentum_weighted = data['momentum_percentile'] * data['volume_momentum_alignment']
    
    # Apply price rejection signals as reversal triggers
    reversal_signal = (data['shadow_ratio'] * data['failed_breakout'] * data['reversal_strength'])
    
    # Combine with persistence factors for signal robustness
    persistence_factor = data['volume_persistence'] * np.sign(data['intraday_momentum'])
    
    # Final composite factor
    composite_factor = (
        -momentum_weighted * 0.4 +  # Negative sign for reversal
        reversal_signal * 0.35 +
        persistence_factor * 0.25
    )
    
    # Clean up and return
    result = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
