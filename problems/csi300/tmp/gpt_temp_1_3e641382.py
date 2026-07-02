import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Price Trend (5-day Linear Regression Slope)
    def rolling_lr_slope(series, window):
        def lr_slope(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        return series.rolling(window=window, min_periods=window).apply(lr_slope)
    
    # Calculate slopes for price and volume trends
    df['short_term_price_slope'] = rolling_lr_slope(df['close'], 5)
    df['medium_term_price_slope'] = rolling_lr_slope(df['close'], 20)
    df['short_term_volume_slope'] = rolling_lr_slope(df['volume'], 5)
    df['medium_term_volume_slope'] = rolling_lr_slope(df['volume'], 20)
    
    # Calculate Price-Volume Divergence
    price_volume_divergence = (
        df['short_term_price_slope'] * df['medium_term_volume_slope'] -
        df['medium_term_price_slope'] * df['short_term_volume_slope']
    )
    
    # Normalize the signal
    price_std = df['close'].rolling(window=20, min_periods=20).std()
    divergence_signal = (price_volume_divergence / price_std) * 100
    
    # Return the divergence signal as a pandas Series
    return divergence_signal
