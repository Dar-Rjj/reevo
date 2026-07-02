import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore, rankdata

def heuristics_v2(df):
    # Calculate Intraday Reversal Signal
    intraday_reversal = (df['high'] - df['low']) / df['close']
    intraday_reversal *= np.sign(df['close'] - df['open'])
    
    # Weight by Normalized Liquidity
    amount_ma20 = df['amount'].rolling(window=20, min_periods=1).mean()
    liquidity_weight = df['amount'] / amount_ma20
    liquidity_reversal = intraday_reversal * liquidity_weight
    
    # Compute Volume Momentum
    volume_ma5 = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_momentum = df['volume'] / volume_ma5
    volume_momentum_rank = volume_momentum.rolling(window=5, min_periods=1).apply(lambda x: rankdata(x)[-1] / len(x))
    
    # Final Reversal Score
    reversal_score = liquidity_reversal * volume_momentum_rank
    
    # Filter Extreme Values
    reversal_score_winsorized = reversal_score.clip(lower=reversal_score.quantile(0.05), upper=reversal_score.quantile(0.95))
    reversal_zscore = zscore(reversal_score_winsorized)
    
    return pd.Series(reversal_zscore, index=df.index)
