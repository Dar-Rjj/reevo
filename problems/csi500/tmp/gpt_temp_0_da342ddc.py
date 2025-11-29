import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Session Gap Momentum Efficiency factor combining gap analysis, volume patterns, 
    and session boundary effects to predict short-term price momentum.
    """
    # Calculate daily metrics
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    df['gap_range_ratio'] = np.abs(df['gap']) / (df['daily_range'] + 1e-8)
    
    # Gap filling patterns
    df['gap_filled'] = np.where(
        (df['gap'] > 0) & (df['low'] <= df['prev_close']), 1,
        np.where((df['gap'] < 0) & (df['high'] >= df['prev_close']), 1, 0)
    )
    
    # Volume-gap alignment
    df['volume_rank'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['gap_volume_alignment'] = np.abs(df['gap']) * df['volume_rank']
    
    # Session boundary momentum
    df['overnight_return'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['momentum_transfer'] = df['overnight_return'] * df['intraday_return']
    
    # Range compression analysis
    df['range_5d'] = df['daily_range'].rolling(window=5, min_periods=3).mean()
    df['range_compression'] = df['daily_range'] / (df['range_5d'] + 1e-8)
    df['compression_volume'] = df['range_compression'] * df['volume_rank']
    
    # Amount flow persistence
    df['amount_flow'] = df['amount'].rolling(window=5, min_periods=3).mean()
    df['flow_efficiency'] = np.abs(df['intraday_return']) / (df['amount_flow'] / df['amount_flow'].rolling(window=20).mean() + 1e-8)
    
    # Closing auction effects
    df['close_open_ratio'] = df['close'] / df['open']
    df['final_hour_signal'] = np.where(
        df['close_open_ratio'] > 1.002, 1,
        np.where(df['close_open_ratio'] < 0.998, -1, 0)
    )
    
    # Combined factor components
    gap_momentum = (
        df['gap_range_ratio'] * 
        (1 - df['gap_filled']) * 
        df['gap_volume_alignment']
    )
    
    boundary_momentum = (
        df['momentum_transfer'] * 
        df['compression_volume'] * 
        df['final_hour_signal']
    )
    
    flow_efficiency = df['flow_efficiency'] * np.sign(df['intraday_return'])
    
    # Final factor combination
    factor = (
        0.4 * gap_momentum +
        0.35 * boundary_momentum +
        0.25 * flow_efficiency
    )
    
    # Normalize and clean
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=60, min_periods=20).mean()) / factor.rolling(window=60, min_periods=20).std()
    
    return factor
