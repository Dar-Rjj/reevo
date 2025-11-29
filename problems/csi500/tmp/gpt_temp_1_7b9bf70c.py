import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling windows
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['volume_volatility'] = df['volume'] * df['high_low_range']
    
    # Rolling calculations
    df['high_low_5d_avg'] = df['high_low_range'].rolling(window=5, min_periods=3).mean()
    df['volume_volatility_10d_avg'] = df['volume_volatility'].rolling(window=10, min_periods=5).mean()
    df['close_3d_avg'] = df['close'].rolling(window=3, min_periods=3).mean()
    df['close_5d_std'] = df['close'].rolling(window=5, min_periods=3).std()
    df['high_low_10d_avg'] = df['high_low_range'].rolling(window=10, min_periods=5).mean()
    df['high_low_20d_avg'] = df['high_low_range'].rolling(window=20, min_periods=10).mean()
    
    # Previous day calculations
    df['prev_close'] = df['close'].shift(1)
    df['prev_2_close'] = df['close'].shift(2)
    
    for i in range(len(df)):
        if i < 20:  # Skip early periods with insufficient data
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        
        # Trend Strength Component
        # Opening Momentum Continuation
        if current['close'] > current['open'] and current['high_low_range'] > 0:
            opening_momentum = current['close_open_diff'] / current['high_low_range']
        else:
            opening_momentum = 0
        
        # Intraday Trend Consistency
        if current['high_low_range'] > 0:
            high_close_diff = current['high'] - current['close']
            close_low_diff = current['close'] - current['low']
            asymmetry = abs(high_close_diff - close_low_diff) / current['high_low_range']
            intraday_consistency = (current['close_open_diff'] / current['high_low_range']) * (1 - asymmetry)
        else:
            intraday_consistency = 0
        
        # Multi-day Trend Acceleration
        if not pd.isna(current['prev_close']) and not pd.isna(current['prev_2_close']):
            two_day_change = current['close'] - current['prev_2_close']
            one_day_change = current['close'] - current['prev_close']
            trend_acceleration = two_day_change - (2 * one_day_change)
        else:
            trend_acceleration = 0
        
        # Volume-Volatility Alignment Component
        # Volatility Expansion
        if current['high_low_5d_avg'] > 0:
            volatility_expansion = current['high_low_range'] / current['high_low_5d_avg']
        else:
            volatility_expansion = 1
        
        # Volume-Volatility Coherence
        if current['volume_volatility_10d_avg'] > 0:
            volume_coherence = current['volume_volatility'] / current['volume_volatility_10d_avg']
        else:
            volume_coherence = 1
        
        # Market Regime Context
        # Trend Stability
        if current['close_5d_std'] > 0:
            trend_stability = 1 - abs(current['close'] - current['close_3d_avg']) / current['close_5d_std']
            trend_stability = max(0, min(1, trend_stability))  # Bound between 0 and 1
        else:
            trend_stability = 1
        
        # Volatility Clustering
        if current['high_low_10d_avg'] > 0 and current['high_low_20d_avg'] > 0:
            volatility_clustering = (current['high_low_range'] / current['high_low_10d_avg']) - (current['high_low_range'] / current['high_low_20d_avg'])
        else:
            volatility_clustering = 0
        
        # Composite Factor Construction
        # Core Factor
        core_factor = opening_momentum * intraday_consistency * trend_acceleration
        
        # Volume-Volatility Enhancement
        volume_enhancement = core_factor * volatility_expansion * volume_coherence
        
        # Final Factor
        final_factor = volume_enhancement * trend_stability * volatility_clustering
        
        result.iloc[i] = final_factor
    
    return result
