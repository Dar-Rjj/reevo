import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate gaps
    df['upper_gap'] = (df['high'] - df['open']) / df['open']
    df['lower_gap'] = (df['open'] - df['low']) / df['open']
    
    # Calculate reversal
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['reversal'] = (df['close'] - df['mid_price']) / df['mid_price']
    
    # Calculate rolling std of reversal (20-day lookback)
    df['reversal_std'] = df['reversal'].rolling(20).std()
    
    # Volume calculations
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    
    for t in range(len(df)):
        if t < 20:  # Skip first 20 days for reliable statistics
            factor.iloc[t] = 0
            continue
            
        current = df.iloc[t]
        
        # Filter extreme moves
        extreme_threshold = 1.5 * current['reversal_std']
        is_extreme = abs(current['reversal']) > extreme_threshold
        
        # Volume filter
        volume_ok = (current['volume'] > current['volume_ma5']) and (current['volume_zscore'] > 0)
        
        # Gap asymmetry
        gap_asymmetry = current['upper_gap'] - current['lower_gap']
        
        # Price confirmation
        price_confirmation = (gap_asymmetry > 0) and (current['reversal'] * gap_asymmetry > 0)
        
        # Composite signal
        if is_extreme and volume_ok and price_confirmation:
            raw_factor = current['reversal'] * gap_asymmetry
        else:
            raw_factor = 0
            
        # Store raw factor for later normalization
        factor.iloc[t] = raw_factor
    
    # Normalization (using expanding window to avoid lookahead)
    ranked = factor.expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    normalized_factor = (ranked - 0.5) * 2  # Scale to [-1, 1]
    
    return normalized_factor
