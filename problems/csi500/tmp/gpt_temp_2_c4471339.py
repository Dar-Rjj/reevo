import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Efficiency Adjustment
    normalized_range = (df['high'] - df['low']) / df['close']
    close_std = df['close'].rolling(5).std()
    price_efficiency = normalized_range / close_std
    
    # Momentum Integration
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    medium_term_momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Liquidity Impact
    volume_sma = df['volume'].rolling(5).mean()
    volume_surge = df['volume'] / volume_sma
    volume_acceleration = (df['volume'] - volume_sma) / volume_sma
    
    # Combined Price Efficiency and Momentum
    combined_signal = price_efficiency * (short_term_momentum + medium_term_momentum)
    
    # Volume Factor Integration
    volume_adjusted_signal = combined_signal * volume_surge
    
    # Apply Directional Bias
    for t in df.index:
        if df.loc[t, 'close'] > df.loc[t, 'open']:
            factor.loc[t] = volume_adjusted_signal.loc[t] * 1  # Long signal
        elif df.loc[t, 'close'] < df.loc[t, 'open']:
            factor.loc[t] = volume_adjusted_signal.loc[t] * -1  # Short signal
        else:
            factor.loc[t] = 0  # Neutral
    
    return factor
