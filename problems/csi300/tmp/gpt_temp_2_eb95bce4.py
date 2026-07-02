import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Compute midpoint (High + Low)/2
    midpoint = (data['high'] + data['low']) / 2
    
    # Compute close vs midpoint
    close_vs_mid = data['close'] - midpoint
    
    # Compute intraday range (High - Low)
    intraday_range = data['high'] - data['low']
    
    # Normalize by intraday range (avoid division by zero)
    reversal_strength = close_vs_mid / intraday_range.replace(0, 1)
    
    # Compute 20-day rolling volume percentile
    volume_percentile = data['volume'].rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Combine signals with volume weighting
    combined_signal = reversal_strength * volume_percentile
    
    # Apply 5-day exponential moving average
    factor = combined_signal.ewm(span=5, adjust=False).mean()
    
    return factor
