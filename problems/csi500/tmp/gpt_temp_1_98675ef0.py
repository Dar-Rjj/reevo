import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Amplitude Compression Breakout Signal
    df['daily_range'] = df['high'] - df['low']
    df['avg_range_5d'] = df['daily_range'].rolling(window=5, min_periods=3).mean()
    df['compression_ratio'] = df['daily_range'] / df['avg_range_5d']
    
    # Compression duration
    compression_mask = df['compression_ratio'] < 0.8
    df['compression_duration'] = compression_mask.groupby(compression_mask.ne(compression_mask.shift()).cumsum()).cumcount() + 1
    df.loc[~compression_mask, 'compression_duration'] = 0
    
    # Volume spike detection
    df['avg_volume_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_spike'] = df['volume'] / df['avg_volume_20d']
    
    # Price confirmation
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['price_trend'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_up'] = df['price_trend'].groupby(df['price_trend'].ne(df['price_trend'].shift()).cumsum()).cumcount() + 1
    df.loc[df['price_trend'] == 0, 'consecutive_up'] = 0
    
    # Amplitude breakout signal
    breakout_condition = (
        (df['compression_duration'] >= 3) & 
        (df['volume_spike'] > 1.5) & 
        (df['close_position'] > 0.6) & 
        (df['consecutive_up'] >= 2)
    )
    amplitude_signal = breakout_condition.astype(float)
    
    # Volume-Weighted Momentum Persistence
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    df['momentum_10d'] = df['close'].pct_change(periods=10)
    
    # Volume percentile rank
    df['volume_rank'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volume-weighted momentum
    df['weighted_momentum'] = (0.6 * df['momentum_3d'] + 0.4 * df['momentum_10d']) * df['volume_rank']
    
    # Momentum persistence
    positive_momentum = df['weighted_momentum'] > 0
    df['consecutive_positive'] = positive_momentum.groupby(positive_momentum.ne(positive_momentum.shift()).cumsum()).cumcount() + 1
    df.loc[~positive_momentum, 'consecutive_positive'] = 0
    
    # Momentum acceleration
    df['momentum_change'] = df['weighted_momentum'] - df['weighted_momentum'].shift(1)
    momentum_signal = df['consecutive_positive'] * (1 + df['momentum_change'].clip(lower=0))
    
    # Intraday Pressure Accumulation
    df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    df['pressure_sum_3d'] = df['net_pressure'].rolling(window=3, min_periods=2).sum()
    
    # Pressure accumulation pattern
    positive_pressure = df['net_pressure'] > 0
    df['consecutive_positive_pressure'] = positive_pressure.groupby(positive_pressure.ne(positive_pressure.shift()).cumsum()).cumcount() + 1
    df.loc[~positive_pressure, 'consecutive_positive_pressure'] = 0
    
    pressure_signal = df['pressure_sum_3d'] * df['consecutive_positive_pressure']
    
    # Range Expansion Momentum
    df['avg_range_10d'] = df['daily_range'].rolling(window=10, min_periods=6).mean()
    df['range_expansion'] = df['daily_range'] / df['avg_range_10d']
    
    expansion_condition = (
        (df['range_expansion'] > 1.3) & 
        (df['close_position'] > 0.5) & 
        (df['volume_spike'] > 1.2)
    )
    
    # Expansion follow-through
    df['next_day_range_ratio'] = df['daily_range'].shift(-1) / df['daily_range']
    df['price_continuation'] = (df['close'].shift(-1) > df['close']).astype(float)
    
    expansion_signal = expansion_condition.astype(float) * df['close_position']
    
    # Volume Clustering Reversal
    high_volume = df['volume_spike'] > 1.8
    df['volume_cluster_duration'] = high_volume.groupby(high_volume.ne(high_volume.shift()).cumsum()).cumcount() + 1
    df.loc[~high_volume, 'volume_cluster_duration'] = 0
    
    # Price action during clusters
    cluster_mask = df['volume_cluster_duration'] > 0
    df['cluster_price_change'] = df['close'].pct_change(periods=1)
    df['extreme_move'] = abs(df['cluster_price_change']) > 0.05
    
    # Reversal signal
    cluster_end = (df['volume_cluster_duration'] == 0) & (df['volume_cluster_duration'].shift(1) >= 3)
    reversal_signal = cluster_end.astype(float) * -df['cluster_price_change'].shift(1)
    
    # Price Elasticity Factor
    df['price_stretch'] = abs(df['close'] - df['open'])
    df['elasticity_ratio'] = df['price_stretch'] / df['daily_range'].replace(0, np.nan)
    df['avg_elasticity_5d'] = df['elasticity_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Elasticity extremes
    high_elasticity = df['elasticity_ratio'] > df['avg_elasticity_5d'] * 1.5
    low_elasticity = df['elasticity_ratio'] < df['avg_elasticity_5d'] * 0.7
    
    elasticity_signal = (
        high_elasticity.astype(float) * -1 + 
        low_elasticity.astype(float) * 0.5
    )
    
    # Combine all signals with weights
    result = (
        0.25 * amplitude_signal +
        0.20 * momentum_signal +
        0.15 * pressure_signal +
        0.15 * expansion_signal +
        0.15 * reversal_signal +
        0.10 * elasticity_signal
    )
    
    # Normalize and clean
    result = (result - result.rolling(window=20, min_periods=10).mean()) / result.rolling(window=20, min_periods=10).std()
    result = result.fillna(0)
    
    return result
