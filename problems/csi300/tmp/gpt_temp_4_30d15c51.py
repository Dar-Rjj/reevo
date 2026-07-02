import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Sentiment Strength branch
    # Normalized rolling mean of volume (window=10)
    volume_rolling_mean = df['volume'].rolling(window=10, min_periods=1).mean()
    normalized_volume = df['volume'] / volume_rolling_mean
    
    # Volatility adjustment (rolling std of close prices, window=10)
    close_rolling_std = df['close'].rolling(window=10, min_periods=1).std()
    volatility_adjustment = 1 / (1 + close_rolling_std)
    
    sentiment_strength = normalized_volume * volatility_adjustment
    
    # Microstructure Imbalance branch
    # Ratio of high to low prices
    price_ratio = df['high'] / df['low']
    
    # Volume confirmation
    volume_delta = df['volume'].diff()  # t - t-1
    amount_rank = df['amount'].rolling(window=5, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_confirmation = volume_delta * amount_rank
    
    microstructure_imbalance = price_ratio * volume_confirmation
    
    # Combine both branches
    factor = sentiment_strength + microstructure_imbalance
    
    return factor
