import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate EMAs
    ema5 = data['close'].ewm(span=5, adjust=False).mean()
    ema20 = data['close'].ewm(span=20, adjust=False).mean()
    
    # Recent Performance Gap
    recent_gap = ema5 - ema20
    rolling_std = data['close'].rolling(window=20).std()
    normalized_gap = recent_gap / rolling_std
    
    # Volume Confirmation
    log_volume = np.log(data['volume'])
    volume_weighted_signal = normalized_gap * log_volume
    
    # Rolling rank of volume-weighted signal
    factor = volume_weighted_signal.rolling(window=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    return factor
