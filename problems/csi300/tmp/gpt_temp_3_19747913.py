import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import zscore

def heuristics_v2(df):
    # Normalized Price Reversal
    price_reversal = (df['close'] - df['open']) / (df['high'] - df['low'])
    price_reversal = np.arcsinh(price_reversal)
    
    # Volume Divergence
    volume_ma = df['volume'].rolling(window=5).mean()
    volume_divergence = df['volume'] / volume_ma
    volume_divergence = np.log(volume_divergence)
    
    # Divergence-Adjusted Reversal
    divergence_adjusted_reversal = price_reversal * volume_divergence
    divergence_adjusted_reversal = divergence_adjusted_reversal.rolling(window=10).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Recent Volatility
    close_std = df['close'].rolling(window=3).std()
    recent_volatility = close_std / df['close'].shift(3)
    
    # Combine with Momentum Confirmation
    factor_values = divergence_adjusted_reversal * recent_volatility
    
    return factor_values
