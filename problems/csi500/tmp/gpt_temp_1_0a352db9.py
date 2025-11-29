import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday returns for different timeframes (assuming 30-min, 1-hour, 2-hour intervals)
    # Since we don't have intraday data, we'll use rolling windows to simulate intraday periods
    data['intraday_return'] = (data['close'] / data['open'] - 1)
    
    # Calculate momentum for different timeframes
    data['momentum_30min'] = data['close'].rolling(window=2).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 2 else np.nan)
    data['momentum_1hr'] = data['close'].rolling(window=4).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 4 else np.nan)
    data['momentum_2hr'] = data['close'].rolling(window=8).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 8 else np.nan)
    
    # Calculate momentum direction and magnitude changes
    data['momentum_30min_dir'] = np.sign(data['momentum_30min'])
    data['momentum_1hr_dir'] = np.sign(data['momentum_1hr'])
    data['momentum_2hr_dir'] = np.sign(data['momentum_2hr'])
    
    # Identify divergence patterns
    data['divergence_30min_1hr'] = data['momentum_30min_dir'] * data['momentum_1hr_dir']
    data['divergence_30min_2hr'] = data['momentum_30min_dir'] * data['momentum_2hr_dir']
    data['divergence_1hr_2hr'] = data['momentum_1hr_dir'] * data['momentum_2hr_dir']
    
    # Calculate divergence strength using momentum ratios
    data['divergence_strength_30min_1hr'] = np.abs(data['momentum_30min'] / (data['momentum_1hr'] + 1e-8))
    data['divergence_strength_30min_2hr'] = np.abs(data['momentum_30min'] / (data['momentum_2hr'] + 1e-8))
    
    # Volume analysis
    data['volume_change'] = data['volume'].pct_change()
    data['volume_momentum'] = data['volume'].rolling(window=5).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 5 else np.nan)
    
    # Volume-price alignment
    data['volume_price_alignment'] = np.sign(data['intraday_return']) * np.sign(data['volume_change'])
    data['volume_acceleration'] = data['volume_change'].rolling(window=3).mean()
    
    # Detect volume-price dislocation
    data['high_momentum_low_volume'] = (np.abs(data['intraday_return']) > data['intraday_return'].rolling(window=20).std()) & \
                                      (data['volume_change'] < data['volume_change'].rolling(window=20).quantile(0.3))
    
    data['volume_spike_no_momentum'] = (data['volume_change'] > data['volume_change'].rolling(window=20).quantile(0.8)) & \
                                      (np.abs(data['intraday_return']) < data['intraday_return'].rolling(window=20).std() * 0.5)
    
    # Momentum quality and persistence
    data['momentum_stability'] = 1 / (data['momentum_30min'].rolling(window=10).std() + 1e-8)
    data['momentum_consistency'] = (data['momentum_30min_dir'].rolling(window=5).sum().abs() / 5)
    
    # Momentum sustainability
    data['momentum_persistence'] = data['momentum_30min_dir'].rolling(window=8).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1]]) / (len(x) - 1) if len(x) > 1 else 0
    )
    
    # Momentum decay patterns
    data['momentum_decay'] = data['momentum_30min'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else np.nan
    )
    
    # Momentum-volume convergence
    data['early_confirmation'] = (data['volume_price_alignment'] > 0) & \
                               (data['momentum_30min_dir'].rolling(window=3).sum().abs() == 3) & \
                               (data['volume_acceleration'] > 0)
    
    data['synchronized_acceleration'] = (np.sign(data['intraday_return']) == np.sign(data['volume_acceleration'])) & \
                                      (np.abs(data['intraday_return']) > data['intraday_return'].rolling(window=20).std()) & \
                                      (np.abs(data['volume_acceleration']) > data['volume_acceleration'].rolling(window=20).std())
    
    # False momentum breakouts
    data['false_breakout'] = data['high_momentum_low_volume'] | data['volume_spike_no_momentum']
    
    # Generate composite factor signal
    # Positive signals
    positive_signals = (
        (data['divergence_30min_1hr'] > 0).astype(int) * 0.2 +
        (data['divergence_30min_2hr'] > 0).astype(int) * 0.15 +
        (data['early_confirmation']).astype(int) * 0.25 +
        (data['synchronized_acceleration']).astype(int) * 0.3 +
        data['momentum_stability'].fillna(0) * 0.1
    )
    
    # Negative signals
    negative_signals = (
        (data['divergence_30min_1hr'] < 0).astype(int) * 0.2 +
        (data['divergence_30min_2hr'] < 0).astype(int) * 0.15 +
        (data['false_breakout']).astype(int) * 0.25 +
        (data['volume_price_alignment'] < 0).astype(int) * 0.3 +
        (1 - data['momentum_consistency'].fillna(0)) * 0.1
    )
    
    # Final composite factor
    data['factor'] = positive_signals - negative_signals
    
    # Apply momentum persistence as weighting
    data['factor'] = data['factor'] * data['momentum_persistence'].fillna(0.5)
    
    # Normalize the factor
    factor_series = data['factor'].copy()
    
    return factor_series
