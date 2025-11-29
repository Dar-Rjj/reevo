import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 20:
        return factor
    
    # High-Low Range Momentum Persistence
    data['high_low_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['high_low_range'].rolling(window=5, min_periods=5).mean()
    data['range_change'] = data['high_low_range'] / data['range_5d_avg'].shift(1)
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    range_momentum = data['range_change'] * data['volume_trend']
    
    # Open-Gap Momentum Divergence
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_return'] = data['close'] / data['open'] - 1
    gap_divergence = data['overnight_gap'] * data['intraday_return'] * np.sign(data['overnight_gap'] + data['intraday_return'])
    
    # Volume-Range Breakout Detector
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_20d_avg']
    range_expansion = data['high_low_range'] / data['range_5d_avg']
    volume_breakout = range_expansion * data['volume_ratio']
    
    # Close-Range Volatility Persistence
    data['range_rank'] = data['high_low_range'].rolling(window=4, min_periods=4).apply(
        lambda x: sum(x.iloc[-1] > x.iloc[:-1]) if len(x) == 4 else np.nan
    )
    volatility_persistence = data['range_rank'] * data['high_low_range'] / data['high_low_range'].rolling(window=10, min_periods=10).mean()
    
    # Intraday Range Pressure Index
    data['buying_pressure'] = (data['close'] - data['low']) / data['high_low_range'].replace(0, np.nan)
    data['selling_pressure'] = (data['high'] - data['close']) / data['high_low_range'].replace(0, np.nan)
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    range_pressure = data['net_pressure'] * data['volume_trend']
    
    # Combine factors with equal weights
    factors_df = pd.DataFrame({
        'range_momentum': range_momentum,
        'gap_divergence': gap_divergence,
        'volume_breakout': volume_breakout,
        'volatility_persistence': volatility_persistence,
        'range_pressure': range_pressure
    })
    
    # Z-score normalization for each factor
    for col in factors_df.columns:
        factors_df[col] = (factors_df[col] - factors_df[col].rolling(window=60, min_periods=20).mean()) / factors_df[col].rolling(window=60, min_periods=20).std()
    
    # Equal-weighted combination
    factor = factors_df.mean(axis=1)
    
    return factor
