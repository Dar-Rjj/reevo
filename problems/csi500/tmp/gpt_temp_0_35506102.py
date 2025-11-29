import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Reversal Signal Component
    # Calculate intraday return momentum
    intraday_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_momentum = intraday_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Compute reversal strength
    prev_close = data['close'].shift(1)
    reversal_strength = np.sign(data['open'] - prev_close) * intraday_momentum
    
    # Apply reversal transformation
    reversal_signal = np.arcsinh(reversal_strength * 100)
    
    # Volume Confirmation Component
    # Calculate volume intensity
    median_volume_20d = data['volume'].rolling(window=20, min_periods=10).median()
    volume_intensity = data['volume'] / median_volume_20d
    volume_intensity = volume_intensity.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Compute volume-price alignment
    volume_price_alignment = np.sign(data['close'] - data['open']) * volume_intensity
    
    # Apply confirmation weighting
    volume_confirmation = np.exp(volume_price_alignment) - 1
    
    # Volatility Regime Component
    # Calculate true range
    high_low = data['high'] - data['low']
    high_prev_close = abs(data['high'] - prev_close)
    low_prev_close = abs(data['low'] - prev_close)
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Compute volatility ratio
    mean_true_range_10d = true_range.rolling(window=10, min_periods=5).mean()
    volatility_ratio = true_range / mean_true_range_10d
    volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Apply volatility scaling
    volatility_scaling = 1 / np.sqrt(volatility_ratio)
    
    # Price Acceleration Component
    # Calculate price velocity
    price_velocity = (data['close'] - data['open']) / (data['high'] - data['low'])
    price_velocity = price_velocity.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Compute acceleration change
    mean_velocity_3d = price_velocity.rolling(window=3, min_periods=2).mean()
    acceleration_change = price_velocity - mean_velocity_3d
    
    # Apply acceleration filter
    acceleration_filter = np.tanh(acceleration_change * 5)
    
    # Final Factor Integration
    # Multiply reversal signal × volume confirmation
    base_factor = reversal_signal * volume_confirmation
    
    # Scale by volatility regime component
    volatility_scaled_factor = base_factor * volatility_scaling
    
    # Add price acceleration component with 0.3 weight
    final_factor = volatility_scaled_factor + (0.3 * acceleration_filter)
    
    # Apply volume-weighted smoothing using 5-day volume rank
    volume_rank_5d = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_weighted_factor = final_factor * volume_rank_5d
    
    return volume_weighted_factor
