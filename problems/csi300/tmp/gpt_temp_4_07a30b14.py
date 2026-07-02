import numpy as np
def heuristics_v2(df):
    # Price Trend Components
    df['intraday_price_change'] = (df['high'] - df['low']) / df['open']
    df['abs_price_change'] = df['intraday_price_change'].abs()
    
    # Rolling standard deviation of price change (10-day window)
    df['price_change_std'] = df['intraday_price_change'].rolling(window=10, min_periods=1).std()
    
    # Normalize the price trend
    df['normalized_price_trend'] = df['abs_price_change'] / (df['price_change_std'] + 1e-6)  # Avoid division by zero
    
    # Volume Trend Components
    # Calculate 20-day median volume (using expanding window to avoid lookahead)
    df['median_volume_20d'] = df['volume'].expanding(min_periods=1).apply(lambda x: x[-20:].median())
    df['volume_deviation'] = df['volume'] / (df['median_volume_20d'] + 1e-6)  # Avoid division by zero
    
    # Combine Signals
    # Multiply normalized price trend by volume deviation
    combined_signal = df['normalized_price_trend'] * df['volume_deviation']
    
    # Scale by sign of price change
    factor = combined_signal * df['intraday_price_change'].apply(np.sign)
    
    return factor
