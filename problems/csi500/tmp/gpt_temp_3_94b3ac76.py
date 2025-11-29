import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume features
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['open_close_range'] = (data['close'] - data['open']) / data['open']
    
    # Calculate hourly momentum breaks (using 4-hour windows for intraday patterns)
    data['hourly_momentum'] = data['open_close_range'].rolling(window=4, min_periods=2).mean()
    data['momentum_accel'] = data['hourly_momentum'].diff()
    data['momentum_gap'] = data['hourly_momentum'].diff().abs()
    
    # Detect fragmentation intensity
    data['momentum_direction'] = np.sign(data['hourly_momentum'])
    data['direction_changes'] = (data['momentum_direction'].diff() != 0).astype(int)
    data['fragmentation_freq'] = data['direction_changes'].rolling(window=6, min_periods=3).mean()
    data['fragmentation_magnitude'] = data['momentum_gap'].rolling(window=6, min_periods=3).mean()
    
    # Fragmentation score combining frequency and magnitude
    data['fragmentation_score'] = (data['fragmentation_freq'] * data['fragmentation_magnitude']).rolling(window=5, min_periods=3).mean()
    
    # Price-volume asymmetry analysis
    data['volume_returns'] = data['volume'] * data['returns']
    data['up_volume'] = np.where(data['returns'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['returns'] < 0, data['volume'], 0)
    
    data['up_volume_ratio'] = data['up_volume'].rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    data['volume_asymmetry'] = (data['up_volume_ratio'] - 0.5).abs()
    
    # Volume support during momentum continuations
    data['momentum_continuation'] = (data['momentum_direction'].shift(1) == data['momentum_direction']).astype(int)
    data['volume_support'] = np.where(data['momentum_continuation'] == 1, 
                                     data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean(), 
                                     0)
    
    # Asymmetry-fragmentation alignment
    data['asymmetry_frag_alignment'] = (data['volume_asymmetry'] * data['fragmentation_score']).rolling(window=5, min_periods=3).mean()
    
    # Session boundary effects (using rolling windows to simulate session patterns)
    data['session_fragmentation'] = data['fragmentation_score'].rolling(window=10, min_periods=5).apply(
        lambda x: x.iloc[:5].mean() - x.iloc[5:].mean() if len(x) == 10 else np.nan, raw=False
    )
    
    data['boundary_asymmetry'] = data['volume_asymmetry'].rolling(window=10, min_periods=5).apply(
        lambda x: x.iloc[5:].std() / (x.iloc[:5].std() + 1e-8), raw=False
    )
    
    # Session consistency
    data['fragmentation_persistence'] = data['fragmentation_score'].rolling(window=8, min_periods=4).std()
    data['asymmetry_stability'] = 1 / (1 + data['volume_asymmetry'].rolling(window=8, min_periods=4).std())
    
    # Final composite factor
    data['composite_factor'] = (
        data['fragmentation_score'] * 
        data['asymmetry_frag_alignment'] * 
        (1 + data['session_fragmentation']) * 
        data['asymmetry_stability'] / 
        (1 + data['fragmentation_persistence'])
    )
    
    # Normalize and clean
    factor = data['composite_factor'].replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    
    return factor
