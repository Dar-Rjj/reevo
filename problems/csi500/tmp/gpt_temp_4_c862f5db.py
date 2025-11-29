import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Compressed Gap Efficiency with Momentum Persistence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price components
    df['prev_close'] = df['close'].shift(1)
    df['gap_magnitude'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Gap persistence calculation
    gap_up_mask = df['gap_magnitude'] > 0
    gap_down_mask = df['gap_magnitude'] < 0
    
    df['gap_persistence'] = 0.0
    df.loc[gap_up_mask, 'gap_persistence'] = (
        (df.loc[gap_up_mask, 'high'] - df.loc[gap_up_mask, 'open']) / 
        (df.loc[gap_up_mask, 'open'] - df.loc[gap_up_mask, 'prev_close'])
    )
    df.loc[gap_down_mask, 'gap_persistence'] = (
        (df.loc[gap_down_mask, 'open'] - df.loc[gap_down_mask, 'low']) / 
        (df.loc[gap_down_mask, 'prev_close'] - df.loc[gap_down_mask, 'open'])
    )
    
    # Intraday gap filling assessment
    df['gap_filling_extent'] = np.where(
        df['gap_magnitude'] > 0,
        (df['close'] - df['open']) / (df['prev_close'] - df['open']),
        (df['close'] - df['open']) / (df['open'] - df['prev_close'])
    )
    
    # True Range calculation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Average True Range (10-day)
    df['atr_10'] = df['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Volatility compression state
    df['vol_compression_ratio'] = df['true_range'] / df['atr_10']
    df['low_vol_state'] = (df['vol_compression_ratio'] < 0.7).astype(int)
    
    # Consecutive low volatility days
    df['consecutive_low_vol'] = df['low_vol_state'].groupby(
        df['low_vol_state'].ne(df['low_vol_state'].shift()).cumsum()
    ).cumcount() + 1
    df.loc[df['low_vol_state'] == 0, 'consecutive_low_vol'] = 0
    
    # Compression intensity
    df['compression_intensity'] = (1 - df['vol_compression_ratio']).clip(0, 1)
    
    # Price efficiency metrics
    df['intraday_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['range_utilization'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Compression boundaries
    df['compression_high'] = df['high'].rolling(window=10, min_periods=5).max()
    df['compression_low'] = df['low'].rolling(window=10, min_periods=5).min()
    
    # Breakout momentum strength
    df['upper_breakout_distance'] = (df['close'] - df['compression_high']) / df['compression_high']
    df['lower_breakout_distance'] = (df['compression_low'] - df['close']) / df['compression_low']
    df['breakout_momentum'] = np.where(
        df['upper_breakout_distance'] > 0,
        df['upper_breakout_distance'],
        -df['lower_breakout_distance']
    )
    
    # Momentum persistence
    df['momentum_1d'] = df['close'].pct_change(1)
    df['momentum_3d'] = df['close'].pct_change(3)
    
    # Volume patterns
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # Calculate alpha components
    for i in range(len(df)):
        if i < 10:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        
        # Volatility-Compressed Gap Efficiency Signal
        if row['consecutive_low_vol'] >= 3 and row['compression_intensity'] > 0.3:
            # High compression environment
            gap_efficiency = (
                row['gap_persistence'] * 0.4 +
                (1 - abs(row['gap_filling_extent'])) * 0.3 +
                row['intraday_efficiency'] * 0.3
            )
        else:
            # Normal or high volatility environment
            gap_efficiency = (
                row['gap_persistence'] * 0.3 +
                row['gap_filling_extent'] * 0.4 +  # Filling is expected in high vol
                row['intraday_efficiency'] * 0.3
            )
        
        # Breakout-Momentum Confirmation Signal
        breakout_strength = (
            np.sign(row['gap_magnitude']) * row['breakout_momentum'] * 0.4 +
            row['momentum_1d'] * 0.3 +
            row['momentum_3d'] * 0.3
        )
        
        # Volume confirmation
        volume_confirmation = np.log1p(row['volume_ratio']) * np.sign(breakout_strength)
        
        # Range expansion dynamics
        range_expansion = (
            (row['high'] - row['low']) / row['atr_10'] * 
            np.sign(row['breakout_momentum'])
        )
        
        # Combined alpha signal
        alpha_signal = (
            gap_efficiency * 0.35 +
            breakout_strength * 0.30 +
            volume_confirmation * 0.20 +
            range_expansion * 0.15
        )
        
        result.iloc[i] = alpha_signal
    
    # Clean up and return
    result = result.fillna(0)
    return result
