import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Expansion Breakout Component
    # Intraday Breakout Assessment
    data['breakout_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['breakout_ratio'] = data['breakout_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout Level vs Rolling Extremes
    data['rolling_high_5'] = data['high'].rolling(window=5, min_periods=1).max()
    data['rolling_low_5'] = data['low'].rolling(window=5, min_periods=1).min()
    data['close_vs_high'] = data['close'] / data['rolling_high_5'] - 1
    data['close_vs_low'] = data['close'] / data['rolling_low_5'] - 1
    
    # Volatility Context Integration
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_10'] = data['daily_range'].rolling(window=10, min_periods=1).mean()
    data['range_expansion'] = data['daily_range'] / data['avg_range_10']
    data['vol_adjusted_breakout'] = data['breakout_ratio'] * data['range_expansion']
    
    # Volume-Regime Acceleration Logic
    # Volume Pressure Assessment
    data['volume_mean_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_mean_10']
    
    # Regime-Based Signal Enhancement
    breakout_signal = data['vol_adjusted_breakout'].copy()
    
    # High Pressure: Invert breakout signal
    high_pressure_mask = data['volume_ratio'] > 1.5
    breakout_signal[high_pressure_mask] = -breakout_signal[high_pressure_mask]
    
    # Normal Pressure: Scale breakout by Volume Ratio
    normal_pressure_mask = (data['volume_ratio'] >= 0.8) & (data['volume_ratio'] <= 1.5)
    breakout_signal[normal_pressure_mask] = breakout_signal[normal_pressure_mask] * data['volume_ratio']
    
    # Low Pressure: Multiply breakout by 1.5
    low_pressure_mask = data['volume_ratio'] < 0.8
    breakout_signal[low_pressure_mask] = breakout_signal[low_pressure_mask] * 1.5
    
    # Price Acceleration Integration
    # Momentum Component
    data['daily_return'] = data['close'] / data['close'].shift(1) - 1
    data['return_avg_5'] = data['daily_return'].rolling(window=5, min_periods=1).mean()
    data['acceleration'] = data['daily_return'] - data['return_avg_5']
    
    # Intraday Position Sensitivity
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['price_position'] = data['price_position'].replace([np.inf, -np.inf], np.nan)
    data['position_adjusted_accel'] = data['acceleration'] * data['price_position']
    
    # Trend Alignment Component
    # Short-term Trend: 5-day Price Slope
    data['trend_slope'] = data['close'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    data['trend_direction'] = np.sign(data['trend_slope'])
    
    # Combine components with trend alignment
    composite_signal = breakout_signal * data['position_adjusted_accel'] * data['trend_direction']
    
    # Final Composite Construction
    # Concentration Confirmation
    data['concentration'] = data['amount'] / data['volume']
    data['concentration'] = data['concentration'].replace([np.inf, -np.inf], np.nan)
    
    # Concentration-Weighted Signal
    concentration_mean = data['concentration'].rolling(window=10, min_periods=1).mean()
    concentration_weight = data['concentration'] / concentration_mean
    final_factor = composite_signal * concentration_weight
    
    return final_factor
