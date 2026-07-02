import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Extremum Signal
    # Rolling rank of close prices over 20 days
    close_rank = df['close'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Normalized absolute 5-day price change
    price_change = df['close'].diff(5).abs()
    norm_price_change = price_change / price_change.rolling(20).std()
    
    # Combine to get Price Extremum Signal
    price_signal = close_rank * norm_price_change
    
    # News Sentiment Confirmation
    if 'sentiment_score' in df.columns:
        # Z-score of sentiment score compared to rolling 10-day mean
        rolling_mean = df['sentiment_score'].rolling(10).mean()
        rolling_std = df['sentiment_score'].rolling(10).std()
        sentiment_zscore = (df['sentiment_score'] - rolling_mean) / rolling_std
        
        # Sign of the Price Extremum Signal
        signal_sign = np.sign(price_signal)
        
        # Combine to get final factor
        factor = signal_sign * sentiment_zscore
    else:
        # If no sentiment data, just use the price signal
        factor = price_signal
    
    return factor
