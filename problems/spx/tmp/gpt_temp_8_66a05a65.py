import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate intraday price volatility
    price_volatility = (df['high'] - df['low']) / df['close']
    
    # Smooth volatility with EWMA (10-day span)
    smoothed_volatility = price_volatility.ewm(span=10, min_periods=10).mean()
    
    # Calculate volume anomalies using rolling MAD (5-day window)
    def rolling_mad(series, window):
        return series.rolling(window).apply(lambda x: (x - x.mean()).abs().mean())
    
    volume_anomalies = rolling_mad(df['volume'], 5)
    
    # Combine smoothed volatility and volume anomalies
    combined_signal = smoothed_volatility * volume_anomalies
    
    return combined_signal
