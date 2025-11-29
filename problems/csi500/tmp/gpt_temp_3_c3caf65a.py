import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Breakout
    # Breakout Ratio: (Close - Low) / (High - Low)
    breakout_ratio = (data['close'] - data['low']) / (data['high'] - data['low'])
    breakout_ratio = breakout_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # 10-day Average Range
    daily_range = data['high'] - data['low']
    avg_range_10d = daily_range.rolling(window=10, min_periods=1).mean()
    
    # Volatility Expansion: (High - Low) / 10-day Average Range
    vol_expansion = daily_range / avg_range_10d
    vol_expansion = vol_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Volume-Regime Signal
    # Volume Pressure: Volume / Volume_MA10
    volume_ma10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_pressure = data['volume'] / volume_ma10
    volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Regime Adjustment
    regime_adjusted_breakout = np.where(
        volume_pressure > 1.5,
        -breakout_ratio,  # Invert when high volume pressure
        breakout_ratio * volume_pressure  # Multiply by pressure otherwise
    )
    
    # Composite Signal
    # Range Expansion: (High - Low) / previous Close
    prev_close = data['close'].shift(1)
    range_expansion = daily_range / prev_close
    range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Final Factor components
    composite_signal = regime_adjusted_breakout * vol_expansion * range_expansion
    
    # Linear regression slope of Close over 5 days
    def linear_regression_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    close_slope = data['close'].rolling(window=5, min_periods=2).apply(
        linear_regression_slope, raw=False
    )
    
    # Final Factor: Composite Signal × sign(slope)
    final_factor = composite_signal * np.sign(close_slope)
    
    return pd.Series(final_factor, index=data.index)
