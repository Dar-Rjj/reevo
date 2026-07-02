import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Gap Detection
    df['Gap'] = df['open'] / df['close'].shift(1) - 1
    df['Absolute Gap'] = df['Gap'].abs()
    
    # Volume Spike Detection
    df['Avg Volume'] = df['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    df['Volume Ratio'] = df['volume'] / df['Avg Volume']
    
    # Combined Signal
    signal = pd.Series(index=df.index)
    for t in df.index:
        if df.loc[t, 'Volume Ratio'] > 1.5:
            if df.loc[t, 'Gap'] > 0:
                signal[t] = df.loc[t, 'Gap'] * df.loc[t, 'Volume Ratio']
            elif df.loc[t, 'Gap'] < 0:
                signal[t] = -1 * df.loc[t, 'Gap'] * df.loc[t, 'Volume Ratio']
        else:
            signal[t] = 0
    
    return signal
