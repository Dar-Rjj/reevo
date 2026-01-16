import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Compute High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 5-day Volume Moving Average
    df['volume_5d_ma'] = df['volume'].rolling(window=5).mean()
    
    # Volume-Adjusted Range
    df['volume_ratio'] = df['volume'] / df['volume_5d_ma']
    df['weighted_range'] = df['high_low_range'] * df['volume_ratio']
    
    # Normalize Intraday Return by Volume-Adjusted Range
    df['momentum_efficiency'] = df['intraday_return'] / df['weighted_range']
    
    # Compute Rolling Volume Slope
    df['volume_slope'] = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / 4)
    
    # Compute Rolling Volume Standard Deviation
    df['volume_std_dev'] = df['volume'].rolling(window=5).std()
    
    # Normalize Volume Slope by Volume Volatility
    df['volume_trend_strength'] = df['volume_slope'] / df['volume_std_dev']
    
    # Compute Rolling Correlation between Close Price and Volume
    df['price_volume_corr'] = df['close'].rolling(window=5).corr(df['volume'])
    
    # Adjust Momentum Efficiency by Volume Trend Strength and Price-Volume Correlation
    df['factor'] = df['momentum_efficiency'] * df['volume_trend_strength'] * df['price_volume_corr']
    
    return df['factor']
