import pandas as pd
import numpy as np
def heuristics_v2(df):
    """
    Calculate Price Momentum with Volume Confirmation factor.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with market data (date index, columns: open, high, low, close, amount, volume)
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    # Calculate Price Momentum
    close_prices = df['close']
    # 20-day rolling window for close prices (using only past data)
    rolling_close = close_prices.rolling(window=20, min_periods=10).mean()
    # Percentile rank (0-1) within 252-day lookback (1 year)
    momentum_rank = close_prices.rolling(window=252, min_periods=63).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    # Confirm with Volume Trend
    volumes = df['volume']
    # 5-day rolling average volume (using only past data)
    rolling_volume = volumes.rolling(window=5, min_periods=3).mean()
    # Check if current volume is above previous day's volume
    volume_increase = (volumes > volumes.shift(1)).astype(int)
    
    # Combine factors
    # Multiply momentum rank by volume confirmation (1 if volume increased, 0 otherwise)
    factor = momentum_rank * volume_increase
    
    return factor
