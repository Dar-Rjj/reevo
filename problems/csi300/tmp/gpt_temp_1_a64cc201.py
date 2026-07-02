import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Divergence Component
    df['Intraday_Divergence'] = (df['high'] - df['low']) / df['open']
    
    df['Close_5days_ago'] = df['close'].shift(5)
    df['Interday_Divergence'] = (df['close'] / df['Close_5days_ago']) - 1
    
    # Reversal Component
    df['Recent_Price_Reversal'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    df['Historical_Reversal'] = df['close'].rolling(window=5).apply(
        lambda x: np.mean((x.iloc[-1] - x.iloc[:-1]) / x.iloc[:-1]), raw=False
    )
    
    # Volume Confirmation
    df['Norm_Volume'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['Volume_Acceleration'] = df['volume'] / df['volume'].shift(1)
    
    # Final Adjustment
    factor = (df['Intraday_Divergence'] + df['Interday_Divergence']) * \
             (df['Recent_Price_Reversal'] + df['Historical_Reversal']) * \
             df['Norm_Volume'] * df['Volume_Acceleration']
    
    return factor.dropna()
