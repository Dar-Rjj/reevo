import numpy as np
def heuristics_v2(df):
    # Calculate intraday momentum (high-low range normalized by close)
    intraday_momentum = (df['high'] - df['low']) / df['close']
    
    # Calculate price volatility (same as momentum in this case)
    price_volatility = (df['high'] - df['low']) / df['close']
    
    # Calculate 5-day volume trend (linear slope)
    volume = df['volume']
    rolling_window = 5
    # Create a function to calculate linear slope for a series
    def linear_slope(series):
        x = np.arange(len(series))
        slope = (len(series) * np.sum(x*series) - np.sum(x)*np.sum(series)) / \
                (len(series)*np.sum(x**2) - np.sum(x)**2)
        return slope
    
    # Calculate rolling slope
    volume_slope = volume.rolling(window=rolling_window).apply(linear_slope, raw=False)
    
    # Combine components
    # Adjust momentum by volatility (divide by volatility to normalize)
    adjusted_momentum = intraday_momentum / (price_volatility + 1e-6)  # Add small constant to avoid division by zero
    
    # Scale by volume signal (current volume multiplied by volume trend slope)
    volume_signal = volume * volume_slope
    
    # Final factor is adjusted momentum multiplied by volume signal
    factor = adjusted_momentum * volume_signal
    
    return factor
