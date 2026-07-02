import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Smoothed Momentum: rolling rank of returns over 20 days
    smoothed_momentum = returns.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Decay Factor: exponential decay with window=10 and decay_rate=0.9
    decay_factor = pd.Series(np.ones(len(df)), index=df.index)
    for i in range(1, len(decay_factor)):
        if i < 10:
            decay_factor.iloc[i] = decay_factor.iloc[i-1] * 0.9
        else:
            decay_factor.iloc[i] = decay_factor.iloc[i-10:i].mean() * 0.9
    
    # Momentum Decay Strength: multiply smoothed momentum by decay factor
    factor = smoothed_momentum * decay_factor
    
    return factor
