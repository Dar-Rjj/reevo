import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Metrics
    df = df.copy()
    df['daily_range'] = df['high'] - df['low']
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['deviation_from_midpoint'] = df['close'] - df['midpoint']
    
    # Assess Range Persistence
    df['range_percentile_rank'] = df['daily_range'].rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5
    )
    
    # Analyze Volume Patterns
    df['volume_5d_avg'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_5d_avg']
    
    # Synthesize Composite Factor
    df['base_reversal'] = -1 * (df['deviation_from_midpoint'] / df['daily_range'])
    df['factor'] = df['base_reversal'] * df['range_percentile_rank'] * df['volume_ratio']
    
    return df['factor']
