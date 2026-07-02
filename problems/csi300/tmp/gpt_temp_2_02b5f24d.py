import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate normalized range
    normalized_range = (data['high'] - data['low']) / data['close']
    
    # Calculate closing position
    closing_position = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Calculate volume distribution
    # Assuming the data is intraday with minute granularity
    morning_volume = data.groupby(data.index.date)['volume'].apply(lambda x: x.iloc[:30].sum())
    afternoon_volume = data.groupby(data.index.date)['volume'].apply(lambda x: x.iloc[-30:].sum())
    
    # Map daily volume back to intraday data
    morning_volume = morning_volume.reindex(data.index.date).ffill()
    afternoon_volume = afternoon_volume.reindex(data.index.date).ffill()
    
    # Calculate volume ratio
    volume_ratio = morning_volume / afternoon_volume
    
    # Calculate efficiency
    efficiency = normalized_range * closing_position
    
    # Calculate rolling 10-day std deviation of close returns
    close_returns = data['close'].pct_change()
    volatility = close_returns.rolling(window=10).std()
    
    # Generate final signal
    final_factor = (efficiency * volume_ratio) / volatility
    
    return final_factor
