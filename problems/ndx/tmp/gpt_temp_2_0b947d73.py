import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize the factor Series with NaN
    factor = pd.Series(index=df.index, dtype=float)
    
    for t in range(len(df)):
        if t < 4:  # Skip the first 4 days due to rolling window requirements
            continue
            
        # Calculate intraday range normalized by close
        intraday_range = (df.iloc[t]['high'] - df.iloc[t]['low']) / df.iloc[t]['close']
        
        # Calculate price reversal signal
        price_reversal_signal = (df.iloc[t]['close'] - df.iloc[t]['open']) / df.iloc[t]['close']
        
        # Apply range adjustment
        range_adjusted_signal = price_reversal_signal / intraday_range
        
        # Compute volume ratio
        volume_ma = df.iloc[t-4:t+1]['volume'].mean()
        volume_ratio = df.iloc[t]['volume'] / volume_ma
        
        # Clip volume ratio between 0.5 and 2.0
        volume_ratio_clipped = max(0.5, min(2.0, volume_ratio))
        
        # Adjust signal by volume ratio
        adjusted_signal = range_adjusted_signal * volume_ratio_clipped
        
        # Compute dollar volume and its 5-day MA
        dollar_volume = df.iloc[t]['volume'] * df.iloc[t]['close']
        dollar_volume_ma = df.iloc[t-4:t+1]['volume'] * df.iloc[t-4:t+1]['close']
        dollar_volume_ma = dollar_volume_ma.mean()
        
        # Apply liquidity threshold
        if dollar_volume > dollar_volume_ma:
            factor.iloc[t] = adjusted_signal
        else:
            factor.iloc[t] = 0
    
    return factor
