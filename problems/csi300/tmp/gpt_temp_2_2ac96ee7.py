import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    # Short-Term Price Trend: 5-day Linear Regression Slope of Close
    df['price_short_trend'] = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
    
    # Medium-Term Price Trend: 20-day Linear Regression Slope of Close
    df['price_medium_trend'] = df['close'].rolling(window=20).apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
    
    # Volume Trend Component
    # Short-Term Volume Trend: 5-day Exponential Moving Average Slope of Volume
    df['vol_short_trend'] = (df['volume'].ewm(span=5, adjust=False).mean().diff())
    
    # Medium-Term Volume Trend: 20-day Exponential Moving Average Slope of Volume
    df['vol_medium_trend'] = (df['volume'].ewm(span=20, adjust=False).mean().diff())
    
    # Divergence Signal
    # Compare Price and Volume Trends
    df['divergence_signal'] = np.where(
        (df['price_short_trend'] > 0) & (df['vol_short_trend'] < 0), -1,
        np.where(
            (df['price_short_trend'] < 0) & (df['vol_short_trend'] > 0), 1, 0
        )
    )
    
    # Volatility Adjustment
    # Multiply by Price Volatility (20d StdDev of Close)
    df['price_volatility'] = df['close'].rolling(window=20).std()
    
    # Divide by Volume Volatility (20d StdDev of Volume)
    df['volume_volatility'] = df['volume'].rolling(window=20).std()
    
    # Final Factor Calculation
    df['factor'] = df['divergence_signal'] * df['price_volatility'] / df['volume_volatility']
    
    return df['factor']
