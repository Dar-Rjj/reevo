import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Momentum Strength
    # Compute (Close - Open) / (High - Low) with zero division protection
    intraday_range = data['high'] - data['low']
    intraday_range = intraday_range.replace(0, np.nan)  # Avoid division by zero
    momentum_strength = (data['close'] - data['open']) / intraday_range
    
    # Compare to 5-day rolling average
    momentum_ratio = momentum_strength / momentum_strength.rolling(window=5, min_periods=3).mean()
    
    # Identify Price-Volume Divergence
    # Calculate |Close - Open| and volume / 10-day average volume
    price_movement = abs(data['close'] - data['open'])
    volume_ratio = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Detect strong price/weak volume or weak price/strong volume
    price_strong = price_movement > price_movement.rolling(window=10, min_periods=5).mean()
    volume_weak = volume_ratio < 1.0
    price_weak = price_movement < price_movement.rolling(window=10, min_periods=5).mean()
    volume_strong = volume_ratio > 1.0
    
    # Divergence multiplier: positive for strong price/weak volume, negative for weak price/strong volume
    divergence_multiplier = np.where(price_strong & volume_weak, 1.2,
                            np.where(price_weak & volume_strong, 0.8, 1.0))
    
    # Evaluate Trend Acceleration
    # Compute 3-day momentum difference
    momentum_3day = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    momentum_2day = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    momentum_acceleration = momentum_3day - momentum_2day
    
    # Compare volume trend to price acceleration
    volume_trend = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    acceleration_aligned = np.sign(momentum_acceleration) == np.sign(volume_trend - 1)
    
    # Acceleration adjustment factor
    acceleration_adjustment = np.where(acceleration_aligned, 1.1, 0.9)
    
    # Generate Composite Factor
    # Score momentum strength (normalized)
    momentum_score = momentum_ratio.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Apply divergence multiplier and acceleration adjustment
    composite_factor = momentum_score * divergence_multiplier * acceleration_adjustment
    
    return pd.Series(composite_factor, index=data.index)
