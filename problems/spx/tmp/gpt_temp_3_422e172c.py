import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate current range
    current_range = df['high'] - df['low']
    
    # Calculate 5-day average range
    historical_range = df['high'].rolling(window=5).apply(lambda x: x.max() - x.min())
    avg_historical_range = historical_range.rolling(window=5).mean()
    
    # Normalize current range by historical range
    normalized_range = current_range / avg_historical_range
    
    # Calculate current volume
    current_volume = df['volume']
    
    # Calculate 5-day average volume
    avg_historical_volume = df['volume'].rolling(window=5).mean()
    
    # Calculate volume spike
    volume_spike = current_volume / avg_historical_volume
    
    # Multiply normalized range by volume spike
    factor = normalized_range * volume_spike
    
    return factor
