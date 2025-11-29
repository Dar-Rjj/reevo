import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Divergence Factor
    Combines multiple momentum signals across trading sessions with volume confirmation
    to detect divergence patterns that may predict future returns.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate session boundaries (assuming 6.5 hour trading day)
    total_minutes = 390  # 6.5 hours
    early_end = 30  # first 30 minutes
    mid_start, mid_end = 30, 120  # 30min to 2 hours
    late_start = 270  # last 2 hours
    
    # Calculate intraday VWAP for early session (using 30-min approximation)
    data['early_vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['early_momentum'] = (data['close'] - data['open']) / data['open'] * 100
    
    # Mid-session momentum (30min to 2hr approximation)
    # Using price change from open to represent mid-session movement
    mid_session_price = (data['high'].rolling(window=3).mean() + data['low'].rolling(window=3).mean()) / 2
    data['mid_momentum'] = (mid_session_price - data['open']) / data['open'] * 100
    
    # Late session momentum (last 2 hours)
    # Using price change from mid-session to close
    late_session_price = (data['high'] + data['low']) / 2
    data['late_momentum'] = (data['close'] - late_session_price) / late_session_price * 100
    
    # Calculate momentum acceleration
    data['momentum_accel'] = data['mid_momentum'] - data['early_momentum']
    
    # Volume patterns
    data['volume_early_ratio'] = data['volume'].rolling(window=5).apply(
        lambda x: x.iloc[0] / x.mean() if x.mean() > 0 else 1
    )
    data['volume_late_ratio'] = data['volume'].rolling(window=5).apply(
        lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1
    )
    
    # Detect divergence patterns
    # Early strong vs late weak
    early_strong = (data['early_momentum'] > data['early_momentum'].rolling(window=10).mean())
    late_weak = (data['late_momentum'] < data['late_momentum'].rolling(window=10).mean())
    divergence_1 = (early_strong & late_weak).astype(int)
    
    # Early weak vs late strong
    early_weak = (data['early_momentum'] < data['early_momentum'].rolling(window=10).mean())
    late_strong = (data['late_momentum'] > data['late_momentum'].rolling(window=10).mean())
    divergence_2 = (early_weak & late_strong).astype(int)
    
    # Mid-session reversal
    mid_reversal = (data['momentum_accel'].abs() > data['momentum_accel'].rolling(window=10).std())
    divergence_3 = mid_reversal.astype(int)
    
    # Magnitude differences
    momentum_gap = (data['early_momentum'] - data['late_momentum']).abs()
    rel_performance = momentum_gap / (data['early_momentum'].abs() + data['late_momentum'].abs() + 1e-8)
    
    # Volume confirmation
    volume_confirmation = (data['volume_early_ratio'] * data['volume_late_ratio'])
    
    # Price-level context
    intraday_range = (data['high'] - data['low']) / data['open']
    high_proximity = (data['high'] - data['close']) / intraday_range
    low_proximity = (data['close'] - data['low']) / intraday_range
    
    # Combine divergence signals with weights
    divergence_score = (
        divergence_1 * 0.3 +
        divergence_2 * 0.3 +
        divergence_3 * 0.2 +
        rel_performance.rank(pct=True) * 0.1 +
        volume_confirmation.rank(pct=True) * 0.1
    )
    
    # Adjust for price context
    price_adjustment = 1 + (high_proximity - low_proximity) * 0.1
    final_factor = divergence_score * price_adjustment
    
    # Normalize and clean
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(window=20).mean()) / final_factor.rolling(window=20).std()
    
    return final_factor
