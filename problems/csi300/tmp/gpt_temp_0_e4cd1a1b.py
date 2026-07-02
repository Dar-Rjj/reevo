import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Assuming 'shares_outstanding' is a column in the DataFrame
    if 'shares_outstanding' not in df.columns:
        df['shares_outstanding'] = df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Calculate Daily Turnover
    df['daily_turnover'] = df['volume'] / df['shares_outstanding']
    
    # Compute 20-day Turnover Moving Average
    df['turnover_ma'] = df['daily_turnover'].rolling(window=20, min_periods=1).mean()
    
    # Normalize Turnover by Historical Avg
    df['normalized_turnover'] = df['daily_turnover'] / df['turnover_ma']
    
    # Calculate Volume Percentile within 20-day window
    df['volume_percentile'] = df['volume'].rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Sigmoid Transformation
    def sigmoid(x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    
    df['volume_weight'] = df['volume_percentile'].apply(sigmoid)
    
    # Combine Signals
    df['factor'] = df['normalized_turnover'] * df['volume_weight']
    
    return df['factor']
