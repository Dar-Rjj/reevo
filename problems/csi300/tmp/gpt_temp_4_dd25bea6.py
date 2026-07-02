import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate 10-day VWAP
    data['vwap_10'] = (data['close'].rolling(window=10).apply(lambda x: (x * data.loc[x.index, 'volume']).sum(), raw=False) /
                       data['volume'].rolling(window=10).sum())
    
    # Calculate Current Range Breakout
    data['range_breakout'] = data['high'] - data['vwap_10']
    
    # Calculate 10-day Volume MA
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    
    # Calculate Volume Surge
    data['volume_surge'] = data['volume'] / data['volume_ma_10']
    
    # Combine Signals
    data['combined_signal'] = data['range_breakout'] * data['volume_surge']
    
    # Scale by 10-day rolling standard deviation of Close
    data['std_close_10'] = data['close'].rolling(window=10).std()
    factor = data['combined_signal'] / data['std_close_10']
    
    return factor.dropna()
