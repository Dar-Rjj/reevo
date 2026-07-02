import pandas as pd
def heuristics_v2(df):
    # Order Flow Imbalance
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Normalize using rolling z-score
    df['close_delta'] = df['close'].diff(1)
    df['rolling_zscore'] = (df['close_delta'] - df['close_delta'].rolling(window=10).mean()) / df['close_delta'].rolling(window=10).std()
    df['order_flow_imbalance'] = df['high_low_ratio'] * df['rolling_zscore']
    
    # Volume Confirmation
    df['volume_rank'] = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volume threshold filter
    df['rolling_mean_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['rolling_mean_volume']
    df['volume_filter'] = df['volume_ratio'].apply(lambda x: 1 if x >= 1.5 else 0)
    
    # Combine factors
    df['microstructure_imbalance'] = df['order_flow_imbalance'] * df['volume_rank'] * df['volume_filter']
    
    return df['microstructure_imbalance']
