import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Improved mean-reversion factor combining volume-confirmed price deviation, volatility scaling, and trend acceleration."""
    # 1. Core mean-reversion signal (20-day return, inverted)
    mean_rev = -df['close'].pct_change(20)
    
    # 2. Volume-confirmed price deviation (current vs 20-day volume-weighted average)
    vol_weighted_avg = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    price_deviation = (df['close'] - vol_weighted_avg) / df['close']
    
    # 3. Volatility scaling (20-day ATR divided by price)
    atr = (df['high'] - df['low']).rolling(20).mean()
    volatility_adj = atr / df['close']
    
    # 4. Trend acceleration (second derivative of 20-day rolling price)
    price_acceleration = df['close'].rolling(20).apply(lambda x: (x[-1] - x[0]) - (x[-2] - x[1]))
    
    # Composite factor: mean-reversion * price deviation / volatility * trend acceleration
    alpha_factor = mean_rev * price_deviation / (volatility_adj + 1e-7) * price_acceleration
    
    return alpha_factor
