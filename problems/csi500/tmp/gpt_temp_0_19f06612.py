import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Adjusted Breakout Momentum
    # Breakout Ratio: (Close - Low) / (High - Low)
    breakout_ratio = (df['close'] - df['low']) / (df['high'] - df['low'])
    breakout_ratio = breakout_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Volatility Expansion: (High - Low) / 10-day Average Range
    range_10d_avg = (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()
    volatility_expansion = (df['high'] - df['low']) / range_10d_avg
    volatility_expansion = volatility_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Volatility-Adjusted Breakout: Breakout Ratio × Volatility Expansion
    volatility_adjusted_breakout = breakout_ratio * volatility_expansion
    
    # Volume-Regime Acceleration & Reversal
    # Volume Pressure: Volume / Volume_MA10
    volume_ma10 = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_pressure = df['volume'] / volume_ma10
    volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Volume-Regime signal
    high_pressure_mask = volume_pressure > 1.5
    volume_regime_signal = pd.Series(index=df.index, dtype=float)
    volume_regime_signal[high_pressure_mask] = -volatility_adjusted_breakout[high_pressure_mask]
    volume_regime_signal[~high_pressure_mask] = volatility_adjusted_breakout[~high_pressure_mask] * volume_pressure[~high_pressure_mask]
    
    # Composite Momentum Factor
    # Range Expansion: (High - Low) / previous Close
    prev_close = df['close'].shift(1).fillna(method='bfill')
    range_expansion = (df['high'] - df['low']) / prev_close
    range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Concentration Level: Amount / Volume
    concentration = df['amount'] / df['volume']
    concentration = concentration.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Composite Signal: Volume-Regime signal × Range Expansion × Concentration
    composite_signal = volume_regime_signal * range_expansion * concentration
    
    # Trend Alignment
    # 5-day Price Slope: Linear regression slope of Close
    def calc_slope(series):
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
    
    price_slope = df['close'].rolling(window=5, min_periods=2).apply(calc_slope, raw=False)
    price_slope = price_slope.fillna(0.0)
    
    # Final Factor: Composite Signal × Price Slope sign
    final_factor = composite_signal * np.sign(price_slope)
    
    return final_factor
