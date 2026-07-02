import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Price Impact Ratio components
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['price_impact_ratio'] = df['price_change'] / df['volume_ma']
    
    # Normalize with cross-sectional rank
    df['price_impact_ratio_rank'] = df['price_impact_ratio'].rank(pct=True)
    
    # Volume Divergence components
    # EMA of volume with decay 0.3
    df['volume_ema'] = df['volume'].ewm(alpha=0.3, adjust=False).mean()
    
    # Price range (delta between high and low)
    df['price_range'] = df['high'] - df['low']
    
    # Correlation between EMA volume and price range
    rolling_corr = []
    for i in range(len(df)):
        if i < 2:  # Need at least 2 points for correlation
            rolling_corr.append(np.nan)
            continue
        window = df.iloc[:i+1]  # Only use data up to current point
        corr = window['volume_ema'].corr(window['price_range'])
        rolling_corr.append(corr)
    df['volume_price_corr'] = rolling_corr
    
    # Z-score of log transformed amount with baseline 0.5
    df['log_amount'] = np.log(df['amount'] + 1e-6)  # Add small constant to avoid log(0)
    mean_log_amount = df['log_amount'].expanding().mean()
    std_log_amount = df['log_amount'].expanding().std()
    df['amount_zscore'] = (df['log_amount'] - mean_log_amount) / (std_log_amount + 1e-6)
    df['amount_zscore'] = df['amount_zscore'] - 0.5  # Subtract baseline
    
    # Combine factors
    df['factor'] = df['price_impact_ratio_rank'] + df['volume_price_corr'].fillna(0) + df['amount_zscore'].fillna(0)
    
    return df['factor']
