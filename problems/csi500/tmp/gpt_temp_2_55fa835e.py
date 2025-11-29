import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Adjusted Breakout
    df['range'] = df['high'] - df['low']
    df['breakout_ratio'] = (df['close'] - df['low']) / df['range']
    df['avg_range_10'] = df['range'].rolling(window=10, min_periods=10).mean()
    df['vol_expansion'] = df['range'] / df['avg_range_10']
    df['vol_adj_breakout'] = df['breakout_ratio'] * df['vol_expansion']
    
    # Volume-Regime Signal
    df['volume_ma10'] = df['volume'].rolling(window=10, min_periods=10).mean()
    df['volume_pressure'] = df['volume'] / df['volume_ma10']
    
    # Apply volume regime logic
    high_pressure_mask = df['volume_pressure'] > 1.5
    df['volume_regime_signal'] = np.where(
        high_pressure_mask,
        -df['vol_adj_breakout'],  # Invert for high pressure
        df['vol_adj_breakout'] * df['volume_pressure']  # Normal pressure
    )
    
    # Composite Signal
    df['prev_close'] = df['close'].shift(1)
    df['range_expansion'] = df['range'] / df['prev_close']
    df['concentration'] = df['amount'] / df['volume']
    df['composite_signal'] = df['volume_regime_signal'] * df['range_expansion'] * df['concentration']
    
    # Price Slope (5-day linear regression)
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
    
    df['price_slope'] = df['close'].rolling(window=5, min_periods=5).apply(calc_slope, raw=False)
    df['slope_sign'] = np.sign(df['price_slope'])
    
    # Final Factor
    df['factor'] = df['composite_signal'] * df['slope_sign']
    
    return df['factor']
