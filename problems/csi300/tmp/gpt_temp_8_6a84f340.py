import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Price Sentiment Strength components
    delta = (df['high'] - df['low']) / df['close']
    volume_factor = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Calculate rolling correlation between delta and volume_factor
    rolling_corr = delta.rolling(window=10).corr(volume_factor)
    
    # Normalize and rank the rolling correlation
    price_sentiment = rolling_corr.rolling(window=20).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    price_sentiment = price_sentiment.rank(pct=True)
    
    # Volatility Dampening components
    volatility = returns.rolling(window=10).std()
    inverse_volatility = 1 / volatility.clip(lower=0.01)
    
    # Apply volatility threshold (0.15 baseline, clip upper at 3.0)
    volatility_threshold = np.minimum(volatility / 0.15, 3.0)
    
    # Combine components
    factor = inverse_volatility * price_sentiment * volatility_threshold
    
    return factor
