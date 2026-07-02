import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate price rejection components
    high_minus_close = data['high'] - data['close']
    close_minus_low = data['close'] - data['low']
    high_minus_low = data['high'] - data['low']
    
    # Avoid division by zero
    high_minus_low = high_minus_low.replace(0, 1)
    
    upper_rejection = high_minus_close / high_minus_low
    lower_rejection = close_minus_low / high_minus_low
    
    # Calculate net rejection
    net_rejection = upper_rejection - lower_rejection
    
    # Calculate volume percentile (rolling 20-day rank)
    volume_rank = data['volume'].rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volume-adjusted rejection
    volume_adjusted_rejection = net_rejection * volume_rank
    
    # Calculate price range stability
    daily_range = (data['high'] - data['low']) / data['close']
    range_stability = daily_range.rolling(5).mean()
    
    # Avoid division by zero
    range_stability = range_stability.replace(0, 1)
    
    # Final factor calculation
    factor = volume_adjusted_rejection / range_stability
    
    return factor
