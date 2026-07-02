import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the input dataframe to avoid modifying the original
    data = df.copy()
    
    # Momentum Divergence Component
    # Relative Strength
    rolling_mean_close = data['close'].rolling(window=10, min_periods=1).mean()
    relative_strength = data['close'] / rolling_mean_close
    
    # Delta of Relative Strength (5 periods)
    delta_relative_strength = relative_strength.diff(5)
    
    # Normalization using rolling standard deviation
    rolling_std_close = data['close'].rolling(window=10, min_periods=1).std()
    normalized_momentum = delta_relative_strength / rolling_std_close
    
    # Liquidity Adjustment
    # Volume-weighted component
    rolling_mean_volume = data['volume'].rolling(window=10, min_periods=1).mean()
    normalized_volume = data['volume'] / rolling_mean_volume
    
    # Spread confirmation
    spread = (data['high'] - data['low']) / data['close']
    volume_rank = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    spread_confirmation = spread * volume_rank
    
    # Microstructure Impact
    # Order flow imbalance
    order_flow_imbalance = np.sign(data['close'] - data['open']) * data['volume'].diff(1)
    
    # EMA smoothing of microstructure impact
    ema_microstructure = order_flow_imbalance.ewm(alpha=0.3, adjust=False).mean()
    
    # Combine all components
    # Volume-weighted momentum
    volume_weighted_momentum = normalized_momentum * normalized_volume
    
    # Final factor with microstructure adjustment
    factor = volume_weighted_momentum * spread_confirmation + ema_microstructure
    
    return factor
