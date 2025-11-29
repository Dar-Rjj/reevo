import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday price reversal components
    # Price Extremes Ratio
    data['price_extremes_ratio'] = (data['high'] - data['open']) / (data['open'] - data['low'])
    # Handle division by zero and infinite values
    data['price_extremes_ratio'] = data['price_extremes_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Price Rejection
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    data['price_rejection'] = (data['close'] - data['open']) / high_low_range
    
    # Combine price reversal signals
    data['raw_reversal'] = data['price_extremes_ratio'] * data['price_rejection']
    
    # Volume confirmation
    # Calculate Volume Intensity
    data['volume_intensity'] = data['volume'] / data['amount']
    data['volume_intensity'] = data['volume_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Adjust reversal signal by volume intensity
    data['volume_adjusted_reversal'] = data['raw_reversal'] * data['volume_intensity']
    
    # Multi-day trend context
    # Calculate short-term trend using 3-day rolling window
    data['price_slope'] = (data['close'] - data['close'].shift(2)) / 2
    
    # Calculate direction consistency (sign consistency over 3 days)
    data['trend_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['trend_consistency'] = data['trend_direction'].rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x == x.iloc[0]) if len(x) > 0 else np.nan, raw=False
    )
    
    # Weight current reversal by trend direction and consistency
    data['trend_weighted_reversal'] = data['volume_adjusted_reversal'] * data['trend_direction'] * data['trend_consistency']
    
    # Volatility filter
    # Calculate rolling volatility (5-day standard deviation of returns)
    data['returns'] = data['close'].pct_change()
    data['rolling_volatility'] = data['returns'].rolling(window=5, min_periods=3).std()
    
    # Calculate Average True Range (ATR) for normalization
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Normalize volatility by ATR
    data['normalized_volatility'] = data['rolling_volatility'] / data['atr']
    data['normalized_volatility'] = data['normalized_volatility'].replace([np.inf, -np.inf], np.nan)
    
    # Apply volatility-based signal enhancement
    # Higher volatility environments get amplified signals
    volatility_multiplier = 1 + data['normalized_volatility'].fillna(0)
    data['final_factor'] = data['trend_weighted_reversal'] * volatility_multiplier
    
    # Clean up intermediate columns
    result = data['final_factor'].copy()
    
    return result
