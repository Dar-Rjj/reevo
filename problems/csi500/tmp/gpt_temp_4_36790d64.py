import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily range
    daily_range = df['high'] - df['low']
    
    # Calculate midpoint (high + low)/2
    midpoint = (df['high'] + df['low']) / 2
    
    # Calculate rolling 5-day median of midpoint
    midpoint_median_5d = midpoint.rolling(window=5, min_periods=1).median()
    
    # Calculate momentum signal
    momentum_signal = midpoint - midpoint_median_5d
    
    # Calculate range percentile rank (current range vs 5-day rolling window)
    range_percentile_rank = daily_range.rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Calculate volume deviation (current volume / rolling 20-day median volume)
    volume_median_20d = df['volume'].rolling(window=20, min_periods=1).median()
    volume_deviation = df['volume'] / volume_median_20d
    
    # Calculate divergence (momentum signal / daily range)
    divergence = momentum_signal / daily_range
    divergence = divergence.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Synthesize composite factor
    factor = divergence * range_percentile_rank * volume_deviation
    
    return factor
