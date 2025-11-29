import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['intraday_high'] = data['high']
    data['intraday_low'] = data['low']
    
    # 1. Intraday Price Reversal Patterns
    
    # Opening Gap Return
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Intraday High-Low Capture
    data['distance_to_high'] = (data['high'] - data['open']) / data['open']
    data['distance_to_low'] = (data['open'] - data['low']) / data['open']
    data['total_intraday_range'] = (data['high'] - data['low']) / data['open']
    
    # Historical context for intraday extremes (20-day rolling window)
    data['rolling_intraday_range'] = data['total_intraday_range'].rolling(window=20, min_periods=10).mean()
    data['rolling_gap_std'] = data['opening_gap'].rolling(window=20, min_periods=10).std()
    
    # Percentile ranks within historical context
    data['range_percentile'] = data['total_intraday_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.8)) if len(x) >= 10 else np.nan, raw=False
    )
    data['gap_percentile'] = data['opening_gap'].abs().rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.8)) if len(x) >= 10 else np.nan, raw=False
    )
    
    # Reversal detection flags
    data['large_gap_limited_follow'] = (
        (data['gap_percentile'] == 1) & 
        (data['range_percentile'] == 0)
    )
    
    data['early_move_reversal'] = (
        (data['distance_to_high'].abs() > data['rolling_intraday_range'] * 0.7) |
        (data['distance_to_low'].abs() > data['rolling_intraday_range'] * 0.7)
    ) & (data['total_intraday_range'] < data['rolling_intraday_range'] * 1.2)
    
    # 2. Volatility Clustering Behavior
    
    # Daily range differences and directional consistency
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['range_change'] = data['daily_range'].diff()
    data['range_direction'] = np.sign(data['range_change'])
    
    # Volatility persistence (5-day window)
    data['volatility_persistence'] = data['range_direction'].rolling(window=5, min_periods=3).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) >= 3 else np.nan, raw=False
    )
    
    # Volatility regime shifts
    data['volatility_break'] = (
        data['daily_range'] > data['daily_range'].rolling(window=20, min_periods=10).quantile(0.9)
    ) | (
        data['daily_range'] < data['daily_range'].rolling(window=20, min_periods=10).quantile(0.1)
    )
    
    # 3. Combine Reversal and Volatility Signals
    
    # Strong signal: Extreme reversal during high volatility clustering
    strong_signal = (
        (data['large_gap_limited_follow'] | data['early_move_reversal']) &
        (data['volatility_persistence'] == 1) &
        (data['volatility_break'] == 1)
    )
    
    # Weak signal: Moderate reversal during stable volatility
    weak_signal = (
        (data['large_gap_limited_follow'] | data['early_move_reversal']) &
        (data['volatility_persistence'] == 1) &
        (data['volatility_break'] == 0)
    )
    
    # Counter signal: No reversal with volatility break
    counter_signal = (
        (~data['large_gap_limited_follow'] & ~data['early_move_reversal']) &
        (data['volatility_break'] == 1)
    )
    
    # Final factor calculation
    factor = pd.Series(index=data.index, dtype=float)
    factor[strong_signal] = 1.0
    factor[weak_signal] = 0.3
    factor[counter_signal] = -0.5
    factor.fillna(0, inplace=True)
    
    return factor
