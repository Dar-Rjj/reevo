import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday components
    daily_range = df['high'] - df['low']
    midpoint = (df['high'] + df['low']) / 2
    
    # Calculate momentum signal using rolling median (5-day window)
    midpoint_rolling_median = midpoint.rolling(window=5, min_periods=1).median()
    momentum_signal = midpoint - midpoint_rolling_median
    
    # Calculate range persistence (percentile rank within 5-day window)
    def percentile_rank(x):
        if len(x) == 1:
            return 0.5
        return (x.rank(pct=True).iloc[-1])
    
    range_percentile_rank = daily_range.rolling(window=5, min_periods=1).apply(percentile_rank, raw=False)
    
    # Calculate volume confirmation components
    extreme_ratio = (df['high'] * df['volume']) / (df['low'] * df['volume'])
    volume_rolling_median = df['volume'].rolling(window=20, min_periods=1).median()
    volume_deviation = df['volume'] / volume_rolling_median
    
    # Synthesize composite factor
    divergence = momentum_signal / daily_range
    divergence = divergence.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Apply range persistence
    factor_with_persistence = divergence * range_percentile_rank
    
    # Apply volume confirmation
    factor_with_volume_confirmation = factor_with_persistence * extreme_ratio
    
    # Final volume adjustment
    final_factor = factor_with_volume_confirmation * volume_deviation
    
    return final_factor
