import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Normalized Spread
    normalized_spread = np.sqrt((df['high'] - df['low']) / df['close'])
    
    # Calculate Volume Contribution
    amount_ma_10 = df['amount'].rolling(window=10, min_periods=1).mean()
    volume_contribution = np.log(df['amount'] / amount_ma_10)
    
    # Combine Price Spread and Volume Contribution
    efficiency_score = normalized_spread * volume_contribution
    
    # Normalize Efficiency Score using z-score
    efficiency_score_normalized = efficiency_score.rolling(window=20, min_periods=1).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Calculate Short-Term Momentum
    momentum_2_day = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Combine Efficiency Score with Momentum
    heuristics_factor = efficiency_score_normalized * momentum_2_day
    
    return pd.Series(heuristics_factor, index=df.index)
