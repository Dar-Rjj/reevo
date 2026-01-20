import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday momentum (High - Low)
    intraday_momentum = df['high'] - df['low']
    
    # Calculate intraday range (High - Low)
    intraday_range = df['high'] - df['low']
    
    # Normalize momentum by intraday range (avoid division by zero)
    normalized_momentum = intraday_momentum / (intraday_range.replace(0, np.nan))
    normalized_momentum = normalized_momentum.fillna(0)
    
    # Compute intraday volatility (5-day rolling std of High-Low)
    intraday_volatility = (df['high'] - df['low']).rolling(5).std()
    
    # Apply asymmetric transformation to volatility
    volatility_adjustment = np.where(
        intraday_volatility > 1.5,
        np.sqrt(intraday_volatility),
        intraday_volatility / 1.5
    )
    
    # Combine components (momentum reversal * volatility adjustment)
    combined_signal = normalized_momentum * volatility_adjustment
    
    # Calculate rolling z-score (3-day window)
    z_score = combined_signal.rolling(3).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if x[:-1].std() != 0 else 0
    )
    
    return z_score
