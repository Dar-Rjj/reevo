import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday momentum
    df['intraday_momentum'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    df['intraday_momentum'] = df['intraday_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate momentum acceleration (change from previous day)
    df['momentum_acceleration'] = df['intraday_momentum'] - df['intraday_momentum'].shift(1)
    
    # Calculate volatility asymmetry
    df['bullish_component'] = np.log(df['high'] / df['open'])
    df['bearish_component'] = np.log(df['low'] / df['open'])
    df['volatility_asymmetry'] = df['bullish_component'] - df['bearish_component']
    
    # Calculate daily range
    df['daily_range'] = df['high'] - df['low']
    
    # Identify momentum exhaustion signals
    df['range_5d_avg'] = df['daily_range'].rolling(window=5, min_periods=3).mean()
    df['high_volatility_flag'] = (df['daily_range'] > 1.2 * df['range_5d_avg']).astype(int)
    
    # Calculate position strength
    df['position_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['position_strength'] = df['position_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate 10-day percentile for position strength
    df['position_percentile'] = df['position_strength'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 5 else np.nan, raw=False
    )
    df['weak_position_flag'] = (df['position_percentile'] < 0.3).astype(int)
    
    # Volume weighting
    df['volume_20d_avg'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_deviation'] = df['volume'] / df['volume_20d_avg'] - 1
    
    # Combine signals with exhaustion filter
    df['filtered_acceleration'] = df['momentum_acceleration'] * (1 - df['high_volatility_flag']) * (1 - df['weak_position_flag'])
    
    # Generate final factor
    df['factor'] = df['filtered_acceleration'] * df['volatility_asymmetry'] * (1 + df['volume_deviation'])
    
    # Apply 3-day smoothing
    df['factor_smoothed'] = df['factor'].rolling(window=3, min_periods=2).mean()
    
    return df['factor_smoothed']
