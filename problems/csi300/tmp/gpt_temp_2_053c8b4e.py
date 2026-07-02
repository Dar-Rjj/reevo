import numpy as np
def heuristics_v2(df):
    """
    Calculate Momentum Acceleration Divergence factor with liquidity adjustment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with columns: ['open', 'high', 'low', 'close', 'volume', 'amount']
        
    Returns
    -------
    pandas.Series
        Factor values indexed by date
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # 1. Rolling Momentum Strength
    # EMA of returns (span=15)
    ema_15 = returns.ewm(span=15, adjust=False).mean()
    # Delta of EMA (span=3)
    momentum_strength = ema_15.diff().ewm(span=3, adjust=False).mean()
    
    # 2. Liquidity Weighted Adjustment
    # Volume ratio: current volume vs 30-day rolling mean
    rolling_volume = df['volume'].rolling(window=30, min_periods=1).mean()
    volume_ratio = df['volume'] / rolling_volume
    
    # Normalize volume ratio cross-sectionally (by date)
    normalized_volume = volume_ratio.groupby(volume_ratio.index).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Combine components
    factor = momentum_strength * normalized_volume
    
    return factor
