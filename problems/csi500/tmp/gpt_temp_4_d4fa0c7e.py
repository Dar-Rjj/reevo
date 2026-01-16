import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Momentum Component
    returns = df['close'].pct_change(5)  # 5-day return
    vol = returns.rolling(window=20).std()  # 20-day rolling std of returns
    momentum = returns / vol  # Normalize by volatility

    # Volume Component
    volume = df['volume']
    volume_slope = volume.rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)

    # Combine Components
    factor = momentum * volume_slope
    factor_normalized = factor / factor.rolling(window=5).mean()

    return factor_normalized
