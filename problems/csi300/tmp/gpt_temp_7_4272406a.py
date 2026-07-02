import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Price Reversal Component
    intraday_position_signal = (df['close'] - df['low']) / (df['high'] - df['low']) - 0.5
    
    # Momentum Adjustment
    momentum_adjustment = df['close'].pct_change(periods=5)
    
    # Volume Confirmation Component
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ratio = df['volume'] / volume_ma_5
    
    # Final Signal Adjustment
    price_component = intraday_position_signal * momentum_adjustment
    final_signal = price_component * volume_ratio
    
    # Apply Z-score normalization
    zscore_final_signal = final_signal.apply(lambda x: (x - final_signal.mean()) / final_signal.std())
    
    return zscore_final_signal
