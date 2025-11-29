import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # High-Low Momentum Divergence
    high_momentum = df['high'] / df['high'].shift(1) - 1
    low_momentum = df['low'] / df['low'].shift(1) - 1
    divergence = high_momentum - low_momentum
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    scaled_divergence = divergence / (daily_range + 1e-8)
    
    # Autocorrelation tracking (5-day)
    autocorr = scaled_divergence.rolling(window=5, min_periods=3).corr(scaled_divergence.shift(1))
    
    # Volume trend (5-day)
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - x[0]) / (np.mean(x) + 1e-8)
    )
    volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    factor1 = scaled_divergence * autocorr * volume_trend * volume_ratio
    
    # Opening Gap Range Efficiency
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_range_current = (df['high'] - df['low']) / df['open']
    gap_range_ratio = gap / (daily_range_current + 1e-8)
    
    # Gap closure speed (how much of the gap is closed by end of day)
    gap_closure = (df['close'] - df['open']) / (gap * df['close'].shift(1) + 1e-8)
    gap_closure_speed = 1 - np.abs(gap_closure)
    
    # Volume acceleration (3-day change in volume momentum)
    volume_accel = df['volume'].pct_change().rolling(window=3, min_periods=2).mean()
    
    factor2 = gap_range_ratio * gap_closure_speed * volume_accel
    
    # Price-Volume Trend Divergence
    price_trend = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - x[0]) / (np.mean(x) + 1e-8)
    )
    volume_trend_pv = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - x[0]) / (np.mean(x) + 1e-8)
    )
    pv_divergence = price_trend - volume_trend_pv
    scaled_pv_divergence = pv_divergence / (np.abs(price_trend) + np.abs(volume_trend_pv) + 1e-8)
    
    # Autocorrelation pattern (3-day)
    pv_autocorr = scaled_pv_divergence.rolling(window=3, min_periods=2).corr(scaled_pv_divergence.shift(1))
    
    # Dollar volume efficiency
    dollar_volume = df['close'] * df['volume']
    dollar_volume_eff = dollar_volume / dollar_volume.rolling(window=20, min_periods=10).mean()
    
    factor3 = scaled_pv_divergence * pv_autocorr * dollar_volume_eff
    
    # Multi-Timeframe Range Momentum
    short_range = (df['high'] - df['low']).rolling(window=5, min_periods=3).mean()
    long_range = (df['high'] - df['low']).rolling(window=20, min_periods=10).mean()
    range_ratio = short_range / (long_range + 1e-8)
    
    # Breakout persistence (days with range expansion)
    range_expansion = (df['high'] - df['low']) > (df['high'] - df['low']).rolling(window=5, min_periods=3).mean()
    breakout_persistence = range_expansion.rolling(window=3, min_periods=2).sum()
    
    factor4 = range_ratio * breakout_persistence * volume_trend
    
    # Close Location Value Momentum
    clv = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    clv_accel = clv.diff().rolling(window=3, min_periods=2).mean()
    
    # CLV persistence (days with same sign CLV)
    clv_sign_persistence = (clv * clv.shift(1) > 0).rolling(window=5, min_periods=3).sum()
    
    factor5 = clv * clv_accel * clv_sign_persistence * volume_accel / (daily_range_current + 1e-8)
    
    # Intraday Volatility Regime Shift
    morning_vol = (df['high'] - df['low']).rolling(window=10, min_periods=5).std()
    afternoon_vol = (df['close'] - df['open']).rolling(window=10, min_periods=5).std()
    vol_ratio = morning_vol / (afternoon_vol + 1e-8)
    
    # Regime shift persistence
    regime_shift = vol_ratio > vol_ratio.rolling(window=10, min_periods=5).mean()
    regime_persistence = regime_shift.rolling(window=5, min_periods=3).sum()
    
    # Volume concentration (intraday volume pattern)
    volume_conc = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    factor6 = vol_ratio * regime_persistence * volume_conc * np.abs(gap)
    
    # Volume-Weighted Price Acceleration
    vwap = (df['high'] + df['low'] + df['close']) / 3
    vwap_accel_short = vwap.pct_change().rolling(window=3, min_periods=2).mean()
    vwap_accel_long = vwap.pct_change().rolling(window=10, min_periods=5).mean()
    vwap_divergence = vwap_accel_short - vwap_accel_long
    
    price_accel = df['close'].pct_change().rolling(window=3, min_periods=2).mean()
    vwap_price_divergence = vwap_accel_short - price_accel
    
    # Momentum consistency
    momentum_consistency = (vwap_accel_short * vwap_accel_short.shift(1) > 0).rolling(window=5, min_periods=3).sum()
    
    intraday_range_eff = (df['high'] - df['low']) / df['open']
    
    factor7 = vwap_divergence * vwap_price_divergence * momentum_consistency * intraday_range_eff
    
    # High-Frequency Reversal Patterns
    price_change = df['close'].pct_change()
    reversal_magnitude = -price_change * price_change.shift(1)
    reversal_frequency = (price_change * price_change.shift(1) < 0).rolling(window=5, min_periods=3).sum()
    
    # Pattern persistence
    pattern_persistence = (reversal_magnitude > reversal_magnitude.rolling(window=10, min_periods=5).mean()
                          ).rolling(window=3, min_periods=2).sum()
    
    volume_spike = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    factor8 = reversal_magnitude * reversal_frequency * pattern_persistence * volume_spike
    
    # Opening Auction Momentum Carry
    pre_open_momentum = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_carry = (df['close'] - df['open']) / (df['open'] + 1e-8)
    momentum_efficiency = momentum_carry / (np.abs(pre_open_momentum) + 1e-8)
    
    # Decay patterns (how momentum decays through the day)
    decay_pattern = 1 - np.abs(momentum_carry / (pre_open_momentum + 1e-8))
    
    opening_volume_conc = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    factor9 = pre_open_momentum * momentum_efficiency * decay_pattern * opening_volume_conc
    
    # Price-Range Compression Breakout
    range_compression = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=20, min_periods=10).mean()
    compression_periods = (range_compression < 0.8).rolling(window=5, min_periods=3).sum()
    
    breakout_strength = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    breakout_direction = np.sign(breakout_strength)
    
    # Follow-through persistence
    follow_through = (breakout_direction * breakout_direction.shift(1) > 0).rolling(window=3, min_periods=2).sum()
    
    volume_expansion = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    factor10 = compression_periods * breakout_strength * breakout_direction * follow_through * volume_expansion
    
    # Combine all factors with equal weights
    combined_factor = (
        factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
        factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
        factor7.fillna(0) + factor8.fillna(0) + factor9.fillna(0) + 
        factor10.fillna(0)
    )
    
    # Normalize the combined factor
    result = (combined_factor - combined_factor.rolling(window=252, min_periods=126).mean()
             ) / (combined_factor.rolling(window=252, min_periods=126).std() + 1e-8)
    
    return result
