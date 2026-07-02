import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Range
    df['range'] = df['high'] - df['low']
    
    # Normalized Range
    df['normalized_range'] = df['range'] / df['close'].shift(1)
    
    # 5-day Moving Average of Normalized Range
    df['ma_normalized_range'] = df['normalized_range'].rolling(window=5, min_periods=1).mean()
    
    # Compression Ratio
    df['compression_ratio'] = df['normalized_range'] / df['ma_normalized_range']
    
    # Volume Percentile
    df['volume_percentile'] = df['volume'].rolling(window=20, min_periods=1).apply(lambda x: (x.rank(pct=True).iloc[-1]))
    
    # Volume Regime
    df['volume_regime'] = df['volume_percentile'].apply(lambda x: 'High Volume' if x > 0.7 else 'Low Volume')
    
    # Factor Construction
    df['factor'] = df.apply(lambda row: -1 * row['compression_ratio'] if row['volume_regime'] == 'High Volume' else row['compression_ratio'], axis=1)
    
    return df['factor']
