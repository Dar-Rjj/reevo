import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Trend Persistence
    df['ema_close'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_close_shifted'] = df['ema_close'].shift(1)  # Using past EMA values
    df['rolling_corr'] = df['close'].rolling(window=15).corr(df['ema_close_shifted'])
    
    # Microstructure Imbalance - Volume Anomaly
    df['rolling_mean_volume'] = df['volume'].rolling(window=20).mean()
    df['zscore_volume'] = (df['volume'] - df['rolling_mean_volume']) / df['volume'].rolling(window=20).std()
    df['normalized_volume'] = df['volume'] / df['volume'].sum()
    
    # Microstructure Imbalance - Spread Deviation
    df['spread'] = df['high'] - df['low']
    df['rolling_rank_spread'] = df['spread'].rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine factors
    df['trend_persistence'] = df['ema_close'] * df['rolling_corr']
    df['volume_anomaly'] = df['zscore_volume'] * df['normalized_volume']
    df['spread_deviation'] = df['spread'] * df['rolling_rank_spread']
    
    df['factor'] = df['trend_persistence'] + df['volume_anomaly'] + df['spread_deviation']
    
    return df['factor']
