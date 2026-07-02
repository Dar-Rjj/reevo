import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Price Trend (5-day Linear Regression Slope)
    def linear_regression_slope(series, window):
        x = np.arange(window)
        def slope(y):
            return np.polyfit(x, y, 1)[0]
        return series.rolling(window).apply(slope)
    
    short_term_price_trend = linear_regression_slope(df['close'], 5)
    
    # Calculate Medium-Term Price Trend (20-day Linear Regression Slope)
    medium_term_price_trend = linear_regression_slope(df['close'], 20)
    
    # Calculate Short-Term Volume Trend (5-day Exponential Moving Average Slope)
    def ema_slope(series, window):
        ema = series.ewm(span=window, adjust=False).mean()
        return ema.diff()
    
    short_term_volume_trend = ema_slope(df['volume'], 5)
    
    # Calculate Medium-Term Volume Trend (20-day Exponential Moving Average Slope)
    medium_term_volume_trend = ema_slope(df['volume'], 20)
    
    # Divergence Signal
    def divergence_signal(price_trend, volume_trend):
        signal = np.where((price_trend > 0) & (volume_trend < 0), -1,
                         np.where((price_trend < 0) & (volume_trend > 0), 1, 0))
        return signal
    
    divergence_signal_short = divergence_signal(short_term_price_trend, short_term_volume_trend)
    divergence_signal_medium = divergence_signal(medium_term_price_trend, medium_term_volume_trend)
    
    # Volatility Adjustment
    price_volatility = df['close'].rolling(20).std()
    volatility_adjusted_signal_short = divergence_signal_short * medium_term_price_trend / price_volatility
    volatility_adjusted_signal_medium = divergence_signal_medium * medium_term_price_trend / price_volatility
    
    # Combine signals
    factor = volatility_adjusted_signal_short + volatility_adjusted_signal_medium
    
    return pd.Series(factor, index=df.index)
