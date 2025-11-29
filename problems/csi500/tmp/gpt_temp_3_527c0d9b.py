import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df = df.copy()
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Intraday Range Efficiency
    intraday_range = df['high'] - df['low']
    efficiency_ratio = intraday_range / true_range
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume Momentum
    volume_median = df['volume'].rolling(window=3, min_periods=1).median()
    volume_momentum = df['volume'] / volume_median
    volume_momentum = volume_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume-Price Divergence
    close_5d_percentile = df['close'].rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    volume_5d_percentile = volume_momentum.rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    divergence_raw = close_5d_percentile - volume_5d_percentile
    
    # Calculate volatility scaling
    returns = df['close'].pct_change()
    volatility_10d = returns.rolling(window=10, min_periods=1).std()
    volatility_10d = volatility_10d.replace(0, np.nan)
    
    divergence_strength = divergence_raw / volatility_10d
    
    # Generate Composite Alpha Factor
    composite_factor = efficiency_ratio * divergence_strength
    
    # Apply conditional sign based on divergence type
    composite_factor = np.where(divergence_raw > 0, composite_factor, -composite_factor)
    
    # Cross-sectional scaling
    final_factor = composite_factor / df['close']
    
    return final_factor
