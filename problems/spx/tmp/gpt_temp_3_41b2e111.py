import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Component
    df['Intraday Range'] = (df['high'] - df['low']) / df['open']
    df['Price Change'] = (df['close'] - df['open']) / df['open']
    
    # Volume Component
    df['MA_Volume_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['Volume Trend'] = df['volume'] / df['MA_Volume_5']
    df['Volume Rank'] = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combined Factor
    df['Raw Momentum'] = (df['Intraday Range'] + df['Price Change']) / 2
    df['Final Factor'] = df['Raw Momentum'] * df['Volume Trend'] / df['Volume Rank']
    
    return df['Final Factor']
