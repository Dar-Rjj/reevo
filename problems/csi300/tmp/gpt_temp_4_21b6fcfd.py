import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Momentum Divergence
    # Calculate EMA ratios
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_ratio = ema5 / ema10
    
    # Calculate rolling rank (cross-sectional)
    rolling_rank = ema_ratio.rolling(window=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    momentum_divergence = rolling_rank
    
    # Liquidity Confirmation
    # Calculate volume ratio
    volume_std = df['volume'].rolling(window=5).std()
    volume_ratio = df['volume'] / (volume_std + 1e-6)  # Add small constant to avoid division by zero
    
    # Calculate z-score of amount
    amount_mean = df['amount'].rolling(window=10).mean()
    amount_std = df['amount'].rolling(window=10).std()
    amount_zscore = (df['amount'] - amount_mean) / (amount_std + 1e-6)
    
    # Combine components
    liquidity_confirmation = volume_ratio * amount_zscore
    
    # Final factor: Price Reversal Signal
    price_reversal = momentum_divergence * liquidity_confirmation
    
    return price_reversal
