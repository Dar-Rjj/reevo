import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Detect Volume Spike
    median_volume = df['volume'].rolling(window=20, min_periods=1).median()
    volume_spike = df['volume'] / median_volume
    
    # Measure Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['open']
    
    # Generate Spike Signal
    spike_signal = volume_spike * intraday_volatility
    
    # Rank by 10-day rolling percentile of Volume
    factor = spike_signal.rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    return factor
