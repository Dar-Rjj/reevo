import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Momentum Divergence
    ema_8 = data['close'].ewm(span=8, adjust=False).mean()
    ema_15 = data['close'].ewm(span=15, adjust=False).mean()
    momentum_divergence = ema_8 / ema_15
    
    # Rolling Rank
    rolling_rank = momentum_divergence.rolling(window=15).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volume Confirmation
    volume_median = data['volume'].rolling(window=15).median()
    volume_ratio = data['volume'] / volume_median
    
    # Decay
    decay = volume_ratio.ewm(alpha=0.3, adjust=False).mean()
    
    # Final Factor Calculation
    factor = rolling_rank * decay
    
    return factor
