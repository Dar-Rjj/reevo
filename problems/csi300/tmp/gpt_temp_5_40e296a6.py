import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Impact branch
    ## Ratio calculation
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_change_ratio'] = df['price_change'] / df['close'].shift(1)
    
    ## Rolling standard deviation of volume
    df['volume_std_10'] = df['volume'].rolling(window=10, min_periods=5).std()
    
    ## Normalize and rank
    df['price_impact'] = df['price_change_ratio'] / (df['volume_std_10'] + 1e-6)
    df['price_impact_rank'] = df['price_impact'].rolling(window=len(df), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volume Divergence branch
    ## EMA of volume with decay=0.3
    df['volume_ema'] = df['volume'].ewm(alpha=0.3, adjust=False).mean()
    
    ## Delta between high and low prices
    df['price_range'] = df['high'] - df['low']
    
    ## Correlation between EMA volume and price range
    df['vol_price_corr'] = df['volume_ema'].rolling(window=20, min_periods=10).corr(df['price_range'])
    
    ## Z-score of log transformed amount
    df['log_amount'] = np.log(df['amount'] + 1e-6)
    df['amount_zscore'] = (df['log_amount'] - df['log_amount'].rolling(window=20, min_periods=10).mean()) / \
                         (df['log_amount'].rolling(window=20, min_periods=10).std() + 1e-6)
    
    # Combine factors
    df['volume_divergence'] = df['vol_price_corr'] * df['amount_zscore']
    
    # Final factor is the sum of both branches
    factor = df['price_impact_rank'] + df['volume_divergence']
    
    return factor
