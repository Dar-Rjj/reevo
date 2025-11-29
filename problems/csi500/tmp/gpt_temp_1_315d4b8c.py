import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Intraday Price Compression Expansion Factor
    # Calculate daily range
    daily_range = (data['high'] - data['low']) / data['close']
    
    # 3-day range change and acceleration
    range_3d_avg = daily_range.rolling(window=3, min_periods=3).mean()
    range_change = daily_range / range_3d_avg
    range_acceleration = range_change.diff()
    
    # Volume-weighted expansion signal
    volume_3d_pct_change = data['volume'].pct_change(periods=3)
    compression_signal = range_change * volume_3d_pct_change
    
    # Volatility regime filtering
    vol_20d = data['close'].pct_change().rolling(window=20, min_periods=20).std()
    vol_regime_strength = vol_20d / vol_20d.rolling(window=20, min_periods=20).mean()
    compression_factor = compression_signal * vol_regime_strength
    
    # Opening Auction Imbalance Momentum
    # Note: Since we only have daily OHLCV, we'll approximate first 30-minute data
    # using the assumption that high/low in first 30 mins is correlated with daily high/low
    open_range_ratio = (data['high'] - data['low']) / data['open']
    prev_day_range = (data['high'].shift(1) - data['low'].shift(1)) / data['open'].shift(1)
    range_expansion = open_range_ratio / prev_day_range
    
    # Approximate first 30-minute close position
    first30_close_pos = (data['close'] - data['low']) / (data['high'] - data['low'])
    directional_bias = first30_close_pos * range_expansion
    
    # Volume confirmation and overnight gap
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=5).mean()
    volume_ratio = data['volume'] / volume_5d_avg
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    auction_factor = directional_bias * volume_ratio * overnight_gap
    
    # Liquidity Absorption Efficiency
    price_impact = abs(data['close'].pct_change()) / data['volume'].replace(0, np.nan)
    efficiency_5d = price_impact.rolling(window=5, min_periods=5).sum()
    efficiency_momentum = efficiency_5d.pct_change(periods=3)
    
    range_adjusted_efficiency = efficiency_5d / daily_range
    volume_acceleration = data['volume'].pct_change(periods=5)
    
    liquidity_signal = efficiency_momentum * range_adjusted_efficiency * volume_acceleration
    liquidity_factor = liquidity_signal * vol_regime_strength
    
    # Price-Level Memory Effect
    recent_high_10d = data['high'].rolling(window=10, min_periods=10).max()
    recent_low_10d = data['low'].rolling(window=10, min_periods=10).min()
    
    # Distance from recent extremes
    dist_from_high = (data['close'] - recent_high_10d) / data['close']
    dist_from_low = (data['close'] - recent_low_10d) / data['close']
    
    # Memory strength calculation
    high_touches = ((data['high'] / recent_high_10d - 1).abs() <= 0.01).rolling(window=10, min_periods=10).sum()
    low_touches = ((data['low'] / recent_low_10d - 1).abs() <= 0.01).rolling(window=10, min_periods=10).sum()
    
    # Exponential decay weights
    decay_weights = np.exp(-np.arange(10) / 3)  # More weight to recent touches
    high_memory = high_touches.rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]) if len(x) == 10 else np.nan
    )
    low_memory = low_touches.rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]) if len(x) == 10 else np.nan
    )
    
    memory_signal = dist_from_high * high_memory - dist_from_low * low_memory
    memory_factor = memory_signal * vol_regime_strength
    
    # Momentum Fractality Dimension
    returns_1d = data['close'].pct_change()
    returns_3d = data['close'].pct_change(periods=3)
    returns_5d = data['close'].pct_change(periods=5)
    
    # Momentum consistency (rolling correlation)
    momentum_consistency = returns_1d.rolling(window=5, min_periods=5).corr(returns_3d)
    
    # Momentum persistence (autocorrelation)
    momentum_persistence = returns_3d.rolling(window=5, min_periods=5).apply(
        lambda x: x.autocorr() if len(x) == 5 else np.nan
    )
    
    volume_momentum = data['volume'].pct_change(periods=5)
    fractality_signal = momentum_consistency * momentum_persistence * volume_momentum
    fractality_factor = fractality_signal * vol_regime_strength
    
    # Volume-Flow Asymmetry Index
    up_days = returns_1d > 0
    down_days = returns_1d < 0
    
    buy_volume_intensity = np.where(up_days, data['volume'] * abs(returns_1d), 0)
    sell_volume_intensity = np.where(down_days, data['volume'] * abs(returns_1d), 0)
    
    buy_volume_5d = pd.Series(buy_volume_intensity, index=data.index).rolling(window=5, min_periods=5).sum()
    sell_volume_5d = pd.Series(sell_volume_intensity, index=data.index).rolling(window=5, min_periods=5).sum()
    
    net_flow = buy_volume_5d - sell_volume_5d
    
    # Flow persistence (consecutive days of net flow direction)
    flow_direction = np.sign(net_flow)
    flow_persistence = flow_direction.rolling(window=5, min_periods=5).apply(
        lambda x: len(x) if len(set(x)) == 1 else 0
    )
    
    range_5d_avg = daily_range.rolling(window=5, min_periods=5).mean()
    flow_signal = net_flow * flow_persistence / range_5d_avg.replace(0, np.nan)
    flow_factor = flow_signal * vol_regime_strength
    
    # Regime-Transition Capture Factor
    vol_5d = data['close'].pct_change().rolling(window=5, min_periods=5).std()
    vol_ratio = vol_5d / vol_20d
    
    # Regime change detection
    vol_regime_change = vol_ratio.diff().abs()
    
    # Momentum regime transition
    returns_5d_avg = returns_5d.rolling(window=5, min_periods=5).mean()
    momentum_cross = (returns_5d_avg * returns_5d_avg.shift(1) < 0).astype(float)
    
    # Volume confirmation during transitions
    volume_avg_20d = data['volume'].rolling(window=20, min_periods=20).mean()
    volume_confirmation = data['volume'] / volume_avg_20d
    
    regime_signal = vol_regime_change * momentum_cross * volume_confirmation
    range_10d_avg = daily_range.rolling(window=10, min_periods=10).mean()
    regime_signal_scaled = regime_signal / range_10d_avg.replace(0, np.nan)
    
    regime_factor = regime_signal_scaled.rolling(window=3, min_periods=3).mean()
    
    # Combine all factors with equal weighting
    factors = [compression_factor, auction_factor, liquidity_factor, 
               memory_factor, fractality_factor, flow_factor, regime_factor]
    
    # Standardize and combine
    combined_factor = pd.Series(0, index=data.index)
    valid_count = pd.Series(0, index=data.index)
    
    for f in factors:
        f_standardized = (f - f.rolling(window=20, min_periods=20).mean()) / f.rolling(window=20, min_periods=20).std()
        combined_factor = combined_factor.add(f_standardized.fillna(0))
        valid_count = valid_count.add(f_standardized.notna().astype(int))
    
    # Average of available factors
    final_factor = combined_factor / valid_count.replace(0, np.nan)
    
    return final_factor
