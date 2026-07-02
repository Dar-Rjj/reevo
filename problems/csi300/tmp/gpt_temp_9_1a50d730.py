import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate returns
    df['return_t'] = df['close'].pct_change()
    
    # Calculate EMA of returns with a span of 10
    df['ema_return'] = df['return_t'].ewm(span=10, min_periods=10).mean()
    
    # Calculate volume divergence
    df['rolling_mean_volume'] = df['volume'].shift(1).rolling(window=20, min_periods=5).mean()
    df['zscore_volume'] = zscore(df['volume'] - df['rolling_mean_volume'], nan_policy='omit')
    
    # Calculate absolute price change
    df['delta_close'] = abs(df['close'] - df['close'].shift(1))
    
    # Calculate volume divergence factor
    df['volume_divergence'] = df['zscore_volume'] * df['delta_close']
    
    # Calculate normalized difference between high and low
    df['normalized_difference'] = (df['high'] - df['low']) / (df['high'] + df['low'])
    
    # Calculate rolling rank of volume over a 15-day window
    df['rolling_rank_volume'] = df['volume'].rolling(window=15, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine factors
    df['liquidity_adjusted_momentum'] = df['ema_return'] + df['volume_divergence']
    df['order_flow_pressure'] = df['normalized_difference'] + df['rolling_rank_volume']
    
    # Final heuristic factor
    df['heuristic_factor'] = df['liquidity_adjusted_momentum'] + df['order_flow_pressure']
    
    return df['heuristic_factor']
