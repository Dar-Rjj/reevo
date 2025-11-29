import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Gap-Decay Momentum with Fractal Volume Acceleration factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Intensity Decay Analysis
    # Short-term Gap Decay
    data['short_gap_decay'] = (data['open'] - data['close'].shift(1)) / \
                             (data['close'].shift(5) - data['close'].shift(10) + 1e-8)
    
    # Medium-term Gap Decay
    data['medium_gap_decay'] = (data['close'].shift(5) - data['close'].shift(10)) / \
                              (data['close'].shift(20) - data['close'].shift(40) + 1e-8)
    
    # Long-term Gap Decay
    data['long_gap_decay'] = (data['close'].shift(20) - data['close'].shift(40)) / \
                            (data['close'].shift(60) - data['close'].shift(120) + 1e-8)
    
    # Gap Magnitude Pattern Recognition
    data['gap_intensity'] = (data['open'] - data['close'].shift(1)).abs()
    data['gap_persistence'] = data['gap_intensity'].rolling(window=5).mean()
    
    # Exponential decay in gap intensity
    data['gap_exp_decay'] = data['gap_intensity'].ewm(span=5).mean()
    
    # Linear decay in gap persistence
    data['gap_linear_decay'] = data['gap_persistence'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else np.nan
    )
    
    # Fractal Volume Acceleration
    # Volume Acceleration
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) / \
                                 (data['volume'].shift(5) - data['volume'].shift(10) + 1e-8)
    
    # Volume Deceleration
    data['volume_deceleration'] = (data['volume'].shift(5) - data['volume'].shift(10)) / \
                                 (data['volume'].shift(20) - data['volume'].shift(40) + 1e-8)
    
    # High-Low-Close volume distribution asymmetry
    data['hlc_volume_asymmetry'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    
    # Price position volume concentration
    data['price_position_volume'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Gap-Volume Decay Divergence
    data['gap_volume_divergence'] = data['short_gap_decay'] - data['volume_acceleration']
    
    # Strong gap decay with weak volume acceleration
    data['strong_gap_weak_volume'] = (data['short_gap_decay'].abs() > data['short_gap_decay'].rolling(20).std()) & \
                                    (data['volume_acceleration'].abs() < data['volume_acceleration'].rolling(20).std())
    
    # Weak gap decay with strong volume acceleration
    data['weak_gap_strong_volume'] = (data['short_gap_decay'].abs() < data['short_gap_decay'].rolling(20).std()) & \
                                    (data['volume_acceleration'].abs() > data['volume_acceleration'].rolling(20).std())
    
    # Decay Pattern Synthesis
    # Multi-timeframe Decay Rate Analysis
    data['short_decay_rate'] = data['short_gap_decay'].pct_change(periods=1)
    data['medium_decay_rate'] = data['medium_gap_decay'].pct_change(periods=5)
    data['long_decay_rate'] = data['long_gap_decay'].pct_change(periods=10)
    
    # Volume Fractal Decay Correlation
    data['volume_gap_correlation'] = data['volume_acceleration'].rolling(window=10).corr(data['short_gap_decay'])
    
    # Generate Gap-Decay Alpha Signal
    # Combine Gap and Volume Decay Signals with weights
    gap_weight = data['gap_intensity'].rolling(window=10).std()
    volume_weight = data['volume'].rolling(window=10).std()
    
    # Main alpha signal components
    data['gap_decay_signal'] = (data['short_gap_decay'] * 0.4 + 
                               data['medium_gap_decay'] * 0.3 + 
                               data['long_gap_decay'] * 0.3)
    
    data['volume_accel_signal'] = (data['volume_acceleration'] * 0.6 + 
                                  data['volume_deceleration'] * 0.4)
    
    # Signal Enhancement Framework
    # Filter for consistent gap-decay patterns
    data['gap_decay_consistency'] = data['gap_decay_signal'].rolling(window=5).std()
    data['volume_consistency'] = data['volume_accel_signal'].rolling(window=5).std()
    
    # Apply fractal volume persistence analysis
    data['volume_fractal_persistence'] = data['price_position_volume'].rolling(window=5).mean()
    
    # Risk adjustment for decay pattern stability
    data['decay_stability'] = 1 / (data['gap_decay_consistency'] + data['volume_consistency'] + 1e-8)
    
    # Timeframe Integration Strategy
    # Intraday patterns (1-5 days)
    intraday_signal = (data['short_gap_decay'] * 0.5 + 
                      data['volume_acceleration'] * 0.3 + 
                      data['hlc_volume_asymmetry'] * 0.2)
    
    # Weekly patterns (5-20 days)
    weekly_signal = (data['medium_gap_decay'] * 0.4 + 
                    data['volume_deceleration'] * 0.3 + 
                    data['gap_volume_divergence'] * 0.3)
    
    # Monthly patterns (20-60 days)
    monthly_signal = (data['long_gap_decay'] * 0.5 + 
                     data['volume_gap_correlation'] * 0.3 + 
                     data['decay_stability'] * 0.2)
    
    # Final alpha signal synthesis
    alpha_signal = (intraday_signal * 0.4 + 
                   weekly_signal * 0.35 + 
                   monthly_signal * 0.25)
    
    # Apply risk adjustment
    final_signal = alpha_signal * data['decay_stability']
    
    # Clean up any infinite values
    final_signal = final_signal.replace([np.inf, -np.inf], np.nan)
    
    return final_signal
