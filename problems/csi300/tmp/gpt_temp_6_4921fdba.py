import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Volatility Compression - Price Stability
    # Calculate rolling standard deviation of close prices over 20 days
    rolling_std = data['close'].rolling(window=20).std()
    
    # Calculate rolling mean of close prices over 10 days
    rolling_mean = data['close'].rolling(window=10).mean()
    
    # Calculate ratio of current close to rolling mean
    ratio = data['close'] / rolling_mean
    
    # Calculate rolling standard deviation of the ratio over 5 days
    ratio_rolling_std = ratio.rolling(window=5).std()
    
    # Normalize the price stability factor using cross-sectional rank
    price_stability = ratio_rolling_std.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Sentiment Confirmation
    # Calculate volume change
    volume_delta = data['volume'] - data['volume'].shift(1)
    
    # Calculate price change
    price_delta = data['close'] - data['close'].shift(1)
    
    # Calculate rolling correlation between volume change and price change (20 days)
    correlation = volume_delta.rolling(window=20).corr(price_delta)
    
    # Calculate z-score of log-transformed amount
    log_amount = np.log1p(data['amount'])
    zscore = (log_amount - log_amount.rolling(window=20).mean()) / log_amount.rolling(window=20).std()
    
    # Combine factors with equal weight
    factor = 0.5 * price_stability + 0.5 * (correlation + zscore)
    
    return factor
