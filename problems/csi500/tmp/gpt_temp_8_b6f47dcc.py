import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Range Breakout Factor
    daily_range = df['high'] - df['low']
    prev_range = daily_range.shift(1)
    range_breakout = (daily_range / prev_range) - 1
    
    # Volume-Adjusted Price Momentum
    price_momentum = df['close'].pct_change(5)
    volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
    vol_adj_momentum = price_momentum * volume_ratio
    
    # Intraday Price Efficiency
    abs_movement = abs(df['high'] - df['low'])
    effective_range = abs(df['open'] - df['close']) + (df['high'] - df['low'])
    efficiency_ratio = 1 - (abs_movement / effective_range)
    
    # Amount-Based Return Persistence
    daily_return = df['close'].pct_change()
    amount_corr = daily_return.rolling(10).corr(df['amount'].rolling(10).mean())
    
    # Opening Gap Volatility Factor
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    close_vol = df['close'].pct_change().rolling(20).std()
    gap_vol_factor = opening_gap / close_vol
    
    # Volume-Weighted Price Reversal
    high_5d = df['high'].rolling(5).max()
    low_5d = df['low'].rolling(5).min()
    dist_to_high = (df['close'] - high_5d) / high_5d
    dist_to_low = (df['close'] - low_5d) / low_5d
    vol_percentile = df['volume'].rolling(20).apply(lambda x: (x[-1] - x.mean()) / x.std())
    vol_weighted_reversal = (dist_to_high + dist_to_low) * vol_percentile
    
    # Intraday Momentum Breakdown
    morning_range = abs(df['open'] - df['high']) + abs(df['open'] - df['low'])
    afternoon_range = abs(df['high'] - df['close'])
    session_ratio = np.log(morning_range / afternoon_range)
    
    # Amount-Volume Divergence Factor
    vwap = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
    vol_return = vwap.pct_change()
    price_return = df['close'].pct_change()
    divergence = abs(price_return - vol_return)
    
    # Close-Relative Position Factor
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Volume-Accelerated Trend Factor
    price_trend = df['close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / x[0])
    vol_acceleration = df['volume'].pct_change(5)
    vol_accel_trend = price_trend * vol_acceleration
    
    # Combine factors (equal weighted)
    factors = pd.DataFrame({
        'range_breakout': range_breakout,
        'vol_adj_momentum': vol_adj_momentum,
        'efficiency': efficiency_ratio,
        'amount_corr': amount_corr,
        'gap_vol': gap_vol_factor,
        'vol_reversal': vol_weighted_reversal,
        'session_ratio': session_ratio,
        'divergence': divergence,
        'close_position': close_position,
        'vol_accel_trend': vol_accel_trend
    })
    
    # Standardize and combine
    factor_combined = factors.apply(lambda x: (x - x.mean()) / x.std()).mean(axis=1)
    
    return factor_combined
