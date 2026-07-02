import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize output series
    signal = pd.Series(0, index=df.index)
    
    # Calculate Overnight Price Gap
    df['gap'] = (df['open'] / df['close'].shift(1)) - 1
    df['abs_gap'] = df['gap'].abs()
    
    # Calculate Volume metrics
    df['avg_volume_20'] = df['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    df['volume_ratio'] = df['volume'] / df['avg_volume_20']
    
    # Calculate Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    
    # Calculate signals
    for t in range(1, len(df)):
        # Skip if missing data
        if pd.isna(df['gap'].iloc[t]) or pd.isna(df['volume_ratio'].iloc[t]) or pd.isna(df['momentum_5'].iloc[t]):
            continue
            
        gap = df['gap'].iloc[t]
        volume_ratio = df['volume_ratio'].iloc[t]
        momentum = df['momentum_5'].iloc[t]
        
        # Base signal based on gap and volume
        if volume_ratio > 1.5:  # High volume condition
            if gap > 0:
                base_signal = gap * volume_ratio
            elif gap < 0:
                base_signal = -1 * gap * volume_ratio
            else:
                base_signal = 0
        else:  # Normal volume
            base_signal = gap
            
        # Adjust signal by momentum
        signal.iloc[t] = base_signal * (1 + momentum)
    
    return signal
