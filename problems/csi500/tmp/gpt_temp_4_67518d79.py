import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Pressure Accumulation Factor
    Combines gap analysis, range compression detection, and pressure-velocity dynamics
    to identify stocks under accumulation pressure.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Recent volatility for gap normalization (20-day rolling)
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['normalized_gap'] = data['overnight_gap'] / (data['volatility_20d'] + 1e-8)
    
    # Intraday range calculations
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['range_ma_5'] = data['daily_range'].rolling(window=5).mean()
    data['range_ratio'] = data['daily_range'] / (data['range_ma_5'] + 1e-8)
    
    # Volume analysis
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma_20'] + 1e-8)
    data['volume_volatility'] = data['volume'].pct_change().rolling(window=10).std()
    
    # Gap absorption metrics (first hour proxy using first 30% of daily range)
    data['first_hour_retrace'] = np.where(
        data['overnight_gap'] > 0,
        (data['low'] - data['open']) / (data['high'] - data['open'] + 1e-8),
        (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    )
    
    # Compression detection
    data['range_contraction'] = data['range_ratio'].rolling(window=3).apply(
        lambda x: 1 if (x.iloc[0] < 0.8 and x.iloc[1] < 0.8 and x.iloc[2] < 0.8) else 0, 
        raw=False
    )
    
    # Volume compression
    data['volume_compression'] = data['volume_volatility'].rolling(window=5).apply(
        lambda x: 1 if x.mean() < x.median() * 0.7 else 0,
        raw=False
    )
    
    # Pressure intensity calculation
    data['compression_intensity'] = (
        (1 - data['range_ratio']) * 
        (1 - data['volume_volatility'] / (data['volume_volatility'].rolling(20).mean() + 1e-8))
    )
    
    # Accumulation duration tracking
    data['compression_duration'] = 0
    for i in range(1, len(data)):
        if data['range_contraction'].iloc[i] == 1 and data['volume_compression'].iloc[i] == 1:
            data['compression_duration'].iloc[i] = data['compression_duration'].iloc[i-1] + 1
    
    # Pressure-velocity dynamics
    data['pressure_gradient'] = data['compression_intensity'].diff(3)
    data['velocity'] = data['pressure_gradient'].rolling(window=5).mean()
    
    # Velocity-price relationship
    data['price_momentum'] = data['close'].pct_change(3)
    data['velocity_divergence'] = data['velocity'] - data['price_momentum']
    
    # Signal generation components
    data['pressure_level'] = data['compression_intensity'] * np.sqrt(data['compression_duration'] + 1)
    data['pressure_velocity'] = data['pressure_level'] * data['velocity']
    
    # Volume confirmation
    data['volume_confirmation'] = np.where(
        (data['volume_ratio'] > 1.2) & (data['velocity'] > 0),
        1.2,
        np.where(
            (data['volume_ratio'] > 0.8) & (data['velocity'] > 0),
            1.0,
            0.8
        )
    )
    
    # Final factor calculation
    data['factor'] = (
        data['pressure_velocity'] * 
        data['volume_confirmation'] * 
        (1 + 0.5 * np.tanh(data['velocity_divergence']))
    )
    
    # Handle edge cases and ensure no lookahead
    data['factor'] = data['factor'].fillna(0)
    
    return data['factor']
