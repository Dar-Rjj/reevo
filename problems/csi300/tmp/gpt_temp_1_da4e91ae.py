import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Trend Strength Component
    # Calculate numerator: rolling sum of (close - open) over 5 days
    numerator = (df['close'] - df['open']).rolling(5).sum()
    
    # Calculate denominator: rolling sum of (high - low) over 5 days
    denominator = (df['high'] - df['low']).rolling(5).sum()
    
    # Calculate ratio (avoid division by zero)
    ratio = numerator / denominator.replace(0, 1)
    
    # Cross-sectional rank and normalize
    trend_strength = ratio.groupby(ratio.index).rank(pct=True)
    
    # Liquidity Confirmation
    # Calculate rolling correlation between volume and close over 10 days
    liquidity_confirmation = df['volume'].rolling(10).corr(df['close'])
    
    # Apply EMA decay with alpha=0.3 and window=5
    def ema_decay(series):
        return series.ewm(alpha=0.3, adjust=False).mean()
    
    decayed_liquidity = liquidity_confirmation.groupby(liquidity_confirmation.index).apply(ema_decay)
    
    # Combine components
    factor = trend_strength * decayed_liquidity
    
    return factor
