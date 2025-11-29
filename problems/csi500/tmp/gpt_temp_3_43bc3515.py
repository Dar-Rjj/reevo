import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['abs_gap'] = np.abs(df['gap'])
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = np.abs(df['high'] - df['prev_close'])
    df['tr3'] = np.abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['avg_tr_5d'] = df['true_range'].rolling(window=5, min_periods=3).mean() / df['prev_close']
    
    # Gap to volatility ratio
    df['gap_vol_ratio'] = df['abs_gap'] / (df['avg_tr_5d'] + 1e-8)
    
    # Intraday retracement strength
    df['high_to_open'] = (df['high'] - df['open']) / df['open']
    df['open_to_low'] = (df['open'] - df['low']) / df['open']
    df['retracement_strength'] = np.where(df['gap'] > 0, df['open_to_low'], df['high_to_open'])
    
    # Volume concentration metrics
    df['morning_volume'] = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x.iloc[-1] / (x.mean() + 1e-8) if len(x) > 0 else np.nan
    )
    
    # Intraday range compression
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    df['avg_range_10d'] = df['daily_range'].rolling(window=10, min_periods=5).mean()
    df['range_compression'] = df['daily_range'] / (df['avg_range_10d'] + 1e-8)
    
    # 5-day high/low distances
    df['high_5d'] = df['high'].rolling(window=5, min_periods=3).max()
    df['low_5d'] = df['low'].rolling(window=5, min_periods=3).min()
    df['dist_to_high'] = (df['high_5d'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df['low_5d']) / df['close']
    
    # Momentum metrics
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.mean() + 1e-8) if len(x) > 0 else np.nan
    )
    
    # Calculate factor components
    # 1. Opening Gap Pressure
    gap_pressure = -df['gap_vol_ratio'] * df['retracement_strength'] * (1 - df['morning_volume'])
    
    # 2. Price Compression Signal
    compression_signal = -df['range_compression'] * (df['dist_to_high'] + df['dist_to_low']) * df['volume_trend']
    
    # 3. Momentum Exhaustion
    momentum_exhaustion = -df['intraday_return'] * df['volume_trend'] * df['range_compression']
    
    # Combine components with weights
    factor = (
        0.4 * gap_pressure +
        0.35 * compression_signal +
        0.25 * momentum_exhaustion
    )
    
    # Normalize and handle missing values
    result = factor.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) > 0 else np.nan
    )
    
    return result
