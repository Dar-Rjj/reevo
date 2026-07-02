import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Short-Term Momentum (EMA with span=5 of returns)
    returns = df['close'].pct_change()
    short_term_momentum = returns.ewm(span=5, adjust=False).mean()
    
    # Cross-sectional rank of momentum
    ranked_momentum = short_term_momentum.groupby(short_term_momentum.index).rank(pct=True)
    
    # Liquidity Adjustment
    # Normalize volume by its 20-day rolling mean
    volume_mean = df['volume'].rolling(window=20, min_periods=1).mean()
    normalized_volume = df['volume'] / volume_mean
    
    # Turnover Score (volume / shares_outstanding if available, otherwise just use volume)
    if 'shares_outstanding' in df.columns:
        turnover_score = df['volume'] / df['shares_outstanding']
    else:
        turnover_score = df['volume']
    
    # Rolling rank of turnover score over 10 days
    ranked_turnover = turnover_score.rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Combine components
    liquidity_adjusted_momentum = ranked_momentum * normalized_volume * ranked_turnover
    
    return liquidity_adjusted_momentum
