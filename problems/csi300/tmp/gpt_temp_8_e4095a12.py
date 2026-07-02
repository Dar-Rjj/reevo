import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Sentiment Strength calculation
    # Normalized rolling mean of volume (window=10)
    rolling_mean_volume = df['volume'].rolling(window=10, min_periods=1).mean()
    normalized_volume = df['volume'] / rolling_mean_volume
    
    # Volatility adjustment using rolling std of close (window=10)
    rolling_std_close = df['close'].rolling(window=10, min_periods=1).std()
    volatility_adjustment = 1 / (1 + rolling_std_close)
    
    sentiment_strength = normalized_volume * volatility_adjustment
    
    # Microstructure Imbalance calculation
    # Ratio of high to low prices
    price_ratio = df['high'] / df['low']
    
    # Volume confirmation
    # Delta of volume (current - previous)
    volume_delta = df['volume'].diff()
    
    # Rolling rank of volume (window=5)
    volume_rank = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    volume_confirmation = volume_delta * volume_rank
    
    microstructure_imbalance = price_ratio * volume_confirmation
    
    # Combine both components
    factor = sentiment_strength + microstructure_imbalance
    
    return factor
