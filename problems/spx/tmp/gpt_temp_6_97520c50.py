import pandas as pd
def heuristics_v2(df):
    # Compute Intraday Momentum
    intraday_momentum = []
    for i in range(len(df)):
        if (df['high'].iloc[i] - df['low'].iloc[i]) > abs(df['close'].iloc[i] - df['open'].iloc[i]):
            momentum = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
        else:
            momentum = (df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i]
        intraday_momentum.append(momentum)
    
    # Calculate 20-day Volume MA (using only past data)
    volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Adjust for Volume Activity
    volume_ratio = df['volume'] / volume_ma
    
    # Combine momentum with volume adjustment
    factor_values = pd.Series(intraday_momentum, index=df.index) * volume_ratio
    
    # Apply Conditional Scaling
    scaled_factor = []
    for i in range(len(df)):
        if df['volume'].iloc[i] > volume_ma.iloc[i]:
            scaled_value = factor_values.iloc[i] * 1.2
        else:
            scaled_value = factor_values.iloc[i] * 0.8
        scaled_factor.append(scaled_value)
    
    return pd.Series(scaled_factor, index=df.index)
