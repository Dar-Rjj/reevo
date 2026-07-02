import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # Ratio of current close to 10-day rolling median of past closes (including current)
    rolling_median = df['close'].rolling(window=10, min_periods=5).median()
    ratio = df['close'] / rolling_median
    
    # Normalize using cross-sectional rank
    price_momentum = ratio.rank(pct=True)
    
    # Liquidity Confirmation
    # Volume delta correlation with amount z-score
    volume_delta = df['volume'] - df['volume'].shift(1)
    amount_zscore = (df['amount'] - df['amount'].rolling(window=5, min_periods=3).mean()) / df['amount'].rolling(window=5, min_periods=3).std()
    
    # Calculate rolling correlation between volume_delta and amount_zscore
    corr_window = 20  # reasonable default window for correlation
    liquidity_corr = volume_delta.rolling(window=corr_window).corr(amount_zscore)
    
    # Decay component - absolute returns with half-life of 10
    returns = df['close'].pct_change()
    decay_factor = np.exp(-np.log(2)/10)  # half-life decay factor
    weights = [decay_factor ** i for i in range(len(df))]
    weights = weights[::-1]  # recent weights first
    decay = returns.abs().rolling(window=len(df), min_periods=1).apply(lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]), raw=True)
    
    # Combine components
    liquidity_confirmation = liquidity_corr * decay
    
    # Final factor is average of normalized components
    factor = (price_momentum + liquidity_confirmation.rank(pct=True)) / 2
    
    return factor
