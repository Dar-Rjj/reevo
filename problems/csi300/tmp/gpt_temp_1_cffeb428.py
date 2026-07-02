import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Short-Term Reversal Signal
    df['delta_close'] = df['close'] - df['close'].shift(1)
    df['reversal_raw'] = -df['delta_close']  # Negative for reversal signal
    df['reversal_rank'] = df['reversal_raw'].rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Liquidity Confirmation
    # Ratio component
    rolling_mean_amount = df['amount'].rolling(window=10).mean()
    df['amount_ratio'] = df['amount'] / rolling_mean_amount
    
    # Z-score component
    delta_amount = df['amount'].diff()
    rolling_std_amount = delta_amount.rolling(window=20).std()
    df['amount_zscore'] = delta_amount / rolling_std_amount
    
    # Combine factors with equal weight
    df['factor'] = 0.5 * df['reversal_rank'] + 0.25 * df['amount_ratio'] + 0.25 * df['amount_zscore']
    
    return df['factor']
