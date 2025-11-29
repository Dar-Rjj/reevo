import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors based on intraday price path complexity and volume dynamics
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic intraday metrics
    data['range'] = data['high'] - data['low']
    data['close_open_diff'] = data['close'] - data['open']
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Price Path Fractality and Momentum Divergence
    # Estimate price path complexity using intraday volatility and range
    data['intraday_volatility'] = data['range'] / data['mid_price']
    data['path_complexity'] = data['intraday_volatility'].rolling(window=5, min_periods=3).std()
    
    # Fractal dimension momentum
    data['fractal_momentum'] = (data['range'] / (data['high'].rolling(window=3).std() + 1e-8)) * data['path_complexity']
    
    # Momentum-path divergence
    data['price_efficiency'] = abs(data['close_open_diff']) / (data['range'] + 1e-8)
    data['direction_consistency'] = np.sign(data['close_open_diff'].rolling(window=3).mean())
    data['momentum_divergence'] = data['close_open_diff'] * data['price_efficiency'] * data['direction_consistency']
    
    # Volume-Price Temporal Asymmetry Patterns
    # Estimate AM/PM volume concentration (using first/last half of day proxy)
    data['volume_concentration'] = data['volume'].rolling(window=5).apply(
        lambda x: x[:len(x)//2].sum() / (x[len(x)//2:].sum() + 1e-8) if len(x) >= 5 else np.nan
    )
    
    # Volume acceleration at extremes
    data['extreme_volume_diff'] = (data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    data['avg_volume'] = data['volume'].rolling(window=10).mean()
    data['volume_acceleration'] = (data['extreme_volume_diff'] / (data['avg_volume'] + 1e-8)) * data['volume'].pct_change(3)
    
    # Gap Filling Efficiency with Volume Confirmation
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = data['open'] - data['prev_close']
    data['gap_filled'] = np.where(
        (data['gap'] > 0) & (data['low'] <= data['prev_close']), 
        -1,  # Gap filled downward
        np.where((data['gap'] < 0) & (data['high'] >= data['prev_close']), 
                1,  # Gap filled upward
                0)   # Gap not filled
    )
    data['gap_filling_momentum'] = data['gap_filled'] * (abs(data['gap']) / (data['range'] + 1e-8)) * data['volume']
    
    # Partial gap rejection
    data['remaining_gap'] = np.where(
        data['gap'] > 0, data['low'] - data['prev_close'],
        np.where(data['gap'] < 0, data['prev_close'] - data['high'], 0)
    )
    data['gap_rejection'] = (abs(data['remaining_gap']) / (abs(data['gap']) + 1e-8)) * data['volume']
    
    # Intraday Regime Transition Detection
    data['range_ratio'] = data['range'] / (data['range'].shift(1) + 1e-8)
    data['regime_persistence'] = data['range_ratio'].rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if abs(x[i] - x[i-1]) < 0.1]) / (len(x) - 1) if len(x) > 1 else 0
    )
    data['volatility_regime_momentum'] = data['range_ratio'] * data['regime_persistence']
    
    # Volume-regime alignment
    data['volume_pattern_change'] = data['volume'].pct_change(3).abs()
    data['price_sensitivity'] = data['close_open_diff'].abs() / (data['range'] + 1e-8)
    data['volume_regime_alignment'] = data['volume_pattern_change'] * data['price_sensitivity'] * data['volatility_regime_momentum']
    
    # Price Compression Breakout with Volume Validation
    data['compression_duration'] = data['range'].rolling(window=10).apply(
        lambda x: len([i for i in range(1, len(x)) if x[i] < x[i-1] * 0.8]) if len(x) >= 10 else 0
    )
    data['breakout_magnitude'] = data['range'] / data['range'].rolling(window=10).mean()
    data['volume_breakout_ratio'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['compression_breakout_efficiency'] = (data['breakout_magnitude'] / (data['compression_duration'] + 1)) * data['volume_breakout_ratio']
    
    # False breakout detection
    data['breakout_reversal'] = np.where(
        (data['close'] < data['open']) & (data['high'] > data['high'].shift(1)),
        -1,
        np.where((data['close'] > data['open']) & (data['low'] < data['low'].shift(1)),
                 1, 0)
    )
    data['volume_divergence'] = np.where(
        data['breakout_reversal'] != 0,
        data['volume'] / data['volume'].rolling(window=5).mean(),
        0
    )
    data['false_breakout_detection'] = data['breakout_reversal'] * data['volume_divergence'] * data['compression_duration'] / 10
    
    # Bidirectional Pressure Accumulation
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8)
    data['volume_weighted_pressure'] = data['buying_pressure'] * (data['volume'] / data['volume'].rolling(window=10).mean())
    
    data['selling_pressure'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    data['volume_decay'] = data['volume'].rolling(window=5).apply(
        lambda x: np.exp(-0.5 * (len(x) - 1)) if len(x) >= 5 else 1
    )
    data['selling_pressure_dissipation'] = data['selling_pressure'] * data['volume_decay']
    
    # Combine all factors with appropriate weights
    components = [
        data['fractal_momentum'] * 0.15,
        data['momentum_divergence'] * 0.12,
        data['volume_concentration'] * 0.10,
        data['volume_acceleration'] * 0.08,
        data['gap_filling_momentum'] * 0.12,
        data['gap_rejection'] * -0.10,  # Negative weight for gap rejection
        data['volatility_regime_momentum'] * 0.09,
        data['volume_regime_alignment'] * 0.08,
        data['compression_breakout_efficiency'] * 0.11,
        data['false_breakout_detection'] * -0.08,  # Negative weight for false breakouts
        data['volume_weighted_pressure'] * 0.07,
        data['selling_pressure_dissipation'] * -0.06  # Negative weight for selling pressure
    ]
    
    # Calculate final factor value
    for i, date in enumerate(data.index):
        valid_components = [comp.iloc[i] for comp in components if not pd.isna(comp.iloc[i])]
        if valid_components:
            factor.iloc[i] = sum(valid_components)
        else:
            factor.iloc[i] = np.nan
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / (factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return factor
