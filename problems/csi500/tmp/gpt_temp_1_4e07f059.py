import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_high_3d'] = data['high'].rolling(window=3, min_periods=3).max()
    data['open_close_diff'] = data['close'] - data['open']
    data['high_close_gap'] = data['high'] - data['close']
    
    # Calculate True Range and Average True Range (20-day)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_20'] = data['true_range'].rolling(window=20, min_periods=20).mean()
    
    # Detect failed breakout conditions
    data['failed_breakout'] = ((data['high'] >= data['prev_high_3d']) & 
                              (data['open_close_diff'] < 0)).astype(int)
    
    # Calculate reversal strength
    data['reversal_strength'] = np.where(
        data['failed_breakout'] == 1,
        data['high_close_gap'] / data['atr_20'],
        0
    )
    
    # Volume analysis
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=10).median()
    data['volume_spike'] = data['volume'] / data['volume_median_10d'] - 1
    data['price_change'] = data['close'] / data['open'] - 1
    
    # Volume-price divergence
    data['abnormal_volume'] = (data['volume_spike'] > 1).astype(int)
    data['volume_price_divergence'] = np.where(
        data['abnormal_volume'] == 1,
        data['volume_spike'] * np.sign(data['price_change']) * -1,  # Negative weighting for divergence
        0
    )
    
    # Combine reversal and divergence signals
    data['raw_signal'] = data['reversal_strength'] * data['volume_price_divergence']
    
    # Market regime adjustment
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=20).std()
    data['volatility_regime'] = data['volatility_20d'] / data['volatility_20d'].rolling(window=60, min_periods=60).mean()
    
    # Apply conditional scaling based on volatility regime
    data['regime_adjusted_signal'] = data['raw_signal'] / (1 + data['volatility_regime'])
    
    # Multi-timeframe confirmation
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_consistency'] = (data['momentum_3d'] * np.sign(data['regime_adjusted_signal']) > 0).astype(int)
    
    # Directional persistence weighting
    data['momentum_weight'] = np.where(
        data['momentum_consistency'] == 1,
        abs(data['momentum_3d']),
        0.5 * abs(data['momentum_3d'])  # Reduced weight for inconsistent signals
    )
    
    # Apply momentum weighting
    data['weighted_signal'] = data['regime_adjusted_signal'] * data['momentum_weight']
    
    # Dynamic signal smoothing with adaptive window
    data['signal_variance'] = data['weighted_signal'].rolling(window=10, min_periods=10).var()
    data['adaptive_window'] = np.where(
        data['signal_variance'] > data['signal_variance'].rolling(window=20, min_periods=20).median(),
        5,  # Shorter window for high volatility
        10   # Longer window for stable periods
    )
    
    # Apply adaptive smoothing
    factor_values = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 20:  # Ensure sufficient data for calculations
            current_window = int(data['adaptive_window'].iloc[i])
            start_idx = max(0, i - current_window + 1)
            factor_values.iloc[i] = data['weighted_signal'].iloc[start_idx:i+1].mean()
    
    # Forward fill any NaN values
    factor_values = factor_values.fillna(method='ffill')
    
    return factor_values
