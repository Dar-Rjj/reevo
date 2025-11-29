import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Core Efficiency-Breakout Components
    # Intraday Efficiency Momentum
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Range Efficiency Trend
    df['efficiency_5d'] = df['intraday_efficiency'].rolling(window=5).mean()
    df['efficiency_10d'] = df['intraday_efficiency'].rolling(window=10).mean()
    df['range_efficiency_trend'] = df['efficiency_5d'] - df['efficiency_10d']
    
    # Efficiency Persistence
    df['efficiency_sign'] = np.sign(df['intraday_efficiency'])
    df['efficiency_persistence'] = df['efficiency_sign'].rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not pd.isna(x.iloc[i]) and not pd.isna(x.iloc[i-1])]), 
        raw=False
    )
    
    # Efficiency Acceleration
    df['efficiency_3d'] = df['intraday_efficiency'].rolling(window=3).mean()
    df['efficiency_acceleration'] = df['efficiency_3d'] - df['efficiency_5d']
    
    # Breakout-Liquidity Integration
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['breakout_up'] = (df['close'] > df['prev_high']).astype(int)
    df['breakout_down'] = (df['close'] < df['prev_low']).astype(int)
    df['breakout_detection'] = df['breakout_up'] - df['breakout_down']
    
    # Liquidity Intensity
    df['liquidity_intensity'] = df['amount'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Breakout Persistence
    df['breakout_persistence'] = df['breakout_detection'].rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    # Volume-Weighted Reversal Divergence
    # Volume Acceleration
    df['prev_volume'] = df['volume'].shift(1)
    df['volume_acceleration'] = (df['volume'] - df['prev_volume']) / df['prev_volume'].replace(0, np.nan)
    
    # Price Deceleration
    df['return_3d'] = df['close'].pct_change(periods=3)
    df['return_5d'] = df['close'].pct_change(periods=5)
    df['price_deceleration'] = df['return_3d'] - df['return_5d']
    
    # Volume-Price Divergence Signal
    df['volume_price_divergence'] = np.sign(df['volume_acceleration']) * np.sign(df['price_deceleration'])
    
    # Gap-Enhanced Reversal Patterns
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['counter_trend_gap'] = df['gap'] * (-df['intraday_efficiency'])
    
    # Extreme Position Rejection
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['extreme_rejection'] = (df['volume'] > df['volume'].rolling(window=20).quantile(0.8)) & (
        (df['range_position'] > 0.8) | (df['range_position'] < 0.2)
    )
    df['extreme_rejection_score'] = df['extreme_rejection'].astype(int)
    
    # Gap Fill Behavior
    df['gap_fill_behavior'] = abs(df['gap']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Multi-timeframe Convergence Analysis
    # Efficiency-Breakout Correlation
    df['efficiency_breakout_corr'] = df['breakout_persistence'] * df['intraday_efficiency']
    
    # Volume Confirmation
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(window=20).mean() * 1.2).astype(int)
    df['volume_confirmation'] = df['volume_surge'] * (df['breakout_detection'] != 0).astype(int)
    
    # Momentum Consistency
    df['return_1d'] = df['close'].pct_change()
    df['momentum_consistency'] = np.sign(df['return_1d']) * np.sign(df['return_3d'])
    
    # Volume-Adaptive Reversal Detection
    df['volume_adaptive_reversal'] = df['volume_acceleration'] * df['price_deceleration']
    
    # Efficiency-Breakout Divergence Patterns
    df['efficiency_breakout_divergence'] = df['intraday_efficiency'] - df['breakout_detection']
    
    # Liquidity-Enhanced Reversal Signals
    df['liquidity_reversal'] = df['liquidity_intensity'] * df['price_deceleration']
    
    # Component Integration
    # Efficiency-Breakout Momentum Score
    df['efficiency_breakout_momentum'] = (
        df['range_efficiency_trend'].fillna(0) + 
        df['efficiency_persistence'].fillna(0) + 
        df['efficiency_acceleration'].fillna(0) +
        df['breakout_detection'].fillna(0) +
        df['breakout_persistence'].fillna(0)
    )
    
    # Volume-Weighted Reversal Divergence Score
    df['volume_reversal_divergence'] = (
        df['volume_price_divergence'].fillna(0) +
        df['counter_trend_gap'].fillna(0) +
        df['extreme_rejection_score'].fillna(0) +
        df['gap_fill_behavior'].fillna(0)
    )
    
    # Multi-timeframe Convergence Score
    df['multi_timeframe_convergence'] = (
        df['efficiency_breakout_corr'].fillna(0) +
        df['volume_confirmation'].fillna(0) +
        df['momentum_consistency'].fillna(0) +
        df['volume_adaptive_reversal'].fillna(0) +
        df['efficiency_breakout_divergence'].fillna(0) +
        df['liquidity_reversal'].fillna(0)
    )
    
    # Signal Weighting and Combination
    # Volume-Adaptive Component Weighting
    volume_weight = df['volume'].rolling(window=20).rank(pct=True)
    
    # Persistence-Based Signal Enhancement
    persistence_weight = df['efficiency_persistence'].rolling(window=5).mean().fillna(0)
    
    # Final Composite Factor
    composite_factor = (
        volume_weight * df['efficiency_breakout_momentum'].fillna(0) +
        (1 + persistence_weight) * df['volume_reversal_divergence'].fillna(0) +
        df['multi_timeframe_convergence'].fillna(0)
    )
    
    result = composite_factor
    
    return result
