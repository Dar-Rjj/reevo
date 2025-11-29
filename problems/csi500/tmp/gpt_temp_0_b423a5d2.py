import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum Signal
    # Compute Midpoint Price Series
    midpoint = (df['high'] + df['low']) / 2
    
    # Calculate Momentum from Open
    momentum = (midpoint - df['open']) / df['open']
    
    # Apply Liquidity Adjustment
    # Calculate Dollar Volume
    dollar_volume = df['close'] * df['volume']
    
    # Compute Liquidity Score
    # Calculate rolling dollar volume percentile (21-day window)
    liquidity_percentile = dollar_volume.rolling(window=21, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Filter signals by liquidity threshold (only keep top 80% liquid stocks)
    liquidity_score = np.where(liquidity_percentile > 0.2, liquidity_percentile, 0)
    
    # Combine Momentum with Liquidity
    # Multiply Momentum by Liquidity Score
    momentum_liquidity = momentum * liquidity_score
    
    # Apply directional consistency check
    # Compare with previous day's momentum
    prev_momentum = momentum.shift(1)
    
    # Enhance persistent moves (same direction as previous day)
    consistency_multiplier = np.where(
        (momentum * prev_momentum) > 0, 
        1.2,  # Boost persistent moves
        1.0   # No change for direction changes
    )
    
    # Output Final Alpha Factor
    alpha_factor = momentum_liquidity * consistency_multiplier
    
    return alpha_factor
