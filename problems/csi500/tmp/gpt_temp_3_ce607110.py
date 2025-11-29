import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Short-term Volatility - 5-day Price Range
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['range_5d'] = data['high_5d'] - data['low_5d']
    
    # Medium-term Volatility - 20-day Price Range
    data['high_20d'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=10).min()
    data['range_20d'] = data['high_20d'] - data['low_20d']
    
    # Long-term Volatility - 40-day Price Range for stability comparison
    data['high_40d'] = data['high'].rolling(window=40, min_periods=20).max()
    data['low_40d'] = data['low'].rolling(window=40, min_periods=20).min()
    data['range_40d'] = data['high_40d'] - data['low_40d']
    
    # Compute Range Ratio for High Volatility Regime
    data['range_ratio'] = data['range_5d'] / (data['range_20d'] + 1e-8)
    
    # Compute Range Stability for Low Volatility Regime
    data['range_stability'] = data['range_20d'] / (data['range_40d'] + 1e-8)
    
    # Regime-Specific Price Patterns
    # High Volatility Pattern
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = data['open'] - data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['intraday_recovery'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['high_vol_pattern'] = data['gap'] * data['intraday_recovery']
    
    # Low Volatility Pattern
    data['close_5d_avg'] = data['close'].rolling(window=5, min_periods=3).mean()
    data['price_compression'] = (data['close'] - data['close_5d_avg']) / (data['range_5d'] + 1e-8)
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_expansion'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    data['low_vol_pattern'] = data['price_compression'] * data['volume_expansion']
    
    # Adaptive Factor Combination
    # Regime Weighting
    # High Volatility Weight based on Range Ratio magnitude and recent gap frequency
    data['range_ratio_rank'] = data['range_ratio'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['gap_freq'] = (data['gap'].abs() > data['gap'].rolling(window=10, min_periods=5).std()).rolling(
        window=5, min_periods=3
    ).mean()
    data['high_vol_weight'] = data['range_ratio_rank'] * data['gap_freq']
    
    # Low Volatility Weight based on Range Stability and recent volume stability
    data['range_stability_rank'] = (1 - data['range_stability']).rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['volume_stability'] = 1 / (data['volume'].rolling(window=10, min_periods=5).std() + 1e-8)
    data['volume_stability_norm'] = data['volume_stability'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['low_vol_weight'] = data['range_stability_rank'] * data['volume_stability_norm']
    
    # Normalize weights to sum to 1
    weight_sum = data['high_vol_weight'] + data['low_vol_weight'] + 1e-8
    data['high_vol_weight_norm'] = data['high_vol_weight'] / weight_sum
    data['low_vol_weight_norm'] = data['low_vol_weight'] / weight_sum
    
    # Signal Integration
    # Weighted Pattern Signals
    data['weighted_high_vol'] = data['high_vol_pattern'] * data['high_vol_weight_norm']
    data['weighted_low_vol'] = data['low_vol_pattern'] * data['low_vol_weight_norm']
    
    # Regime Transition Adjustment
    data['regime_change'] = data['range_ratio'].diff(3).abs()
    data['recent_regime_changes'] = data['regime_change'].rolling(window=5, min_periods=3).mean()
    data['transition_adjustment'] = 1 / (1 + data['recent_regime_changes'])
    
    # Final factor with smoothing
    data['raw_factor'] = (data['weighted_high_vol'] + data['weighted_low_vol']) * data['transition_adjustment']
    data['factor'] = data['raw_factor'].rolling(window=3, min_periods=2).mean()
    
    return data['factor']
