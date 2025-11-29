import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel intraday alpha factors with fragmentation and efficiency concepts
    """
    data = df.copy()
    
    # Calculate basic metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['overnight_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # 1. Intraday Reversal Asymmetry with Fragmentation
    # Fragmented morning reversal
    morning_reversal_numerator = data['open'] - data['low']
    morning_reversal_denominator = data['high'] - data['open']
    morning_reversal_denominator = np.where(morning_reversal_denominator == 0, 1e-6, morning_reversal_denominator)
    morning_reversal = morning_reversal_numerator / morning_reversal_denominator
    
    # Calculate intraday direction changes (proxy for fragmentation)
    price_changes = data['close'].diff()
    direction_changes = ((price_changes > 0) & (price_changes.shift(1) < 0)) | ((price_changes < 0) & (price_changes.shift(1) > 0))
    intraday_direction_changes = direction_changes.rolling(window=5, min_periods=3).sum()
    
    fragmented_morning_reversal = morning_reversal * intraday_direction_changes
    
    # Dispersed afternoon reversal
    afternoon_reversal_numerator = data['high'] - data['close']
    afternoon_reversal_denominator = data['close'] - data['low']
    afternoon_reversal_denominator = np.where(afternoon_reversal_denominator == 0, 1e-6, afternoon_reversal_denominator)
    afternoon_reversal = afternoon_reversal_numerator / afternoon_reversal_denominator
    
    # Hourly return variance (proxy using intraday volatility)
    intraday_range = (data['high'] - data['low']) / data['open']
    hourly_variance = intraday_range.rolling(window=5, min_periods=3).std()
    
    dispersed_afternoon_reversal = afternoon_reversal * hourly_variance
    
    # 2. Volume-Weighted Price Efficiency Extremes
    # High-volume price efficiency
    price_range = data['high'] - data['low']
    vwap_proxy = data['amount'] / np.where(data['volume'] == 0, 1e-6, data['volume'])
    price_efficiency = price_range / np.where(vwap_proxy == 0, 1e-6, vwap_proxy)
    
    returns_per_volume = data['intraday_return'] / np.where(data['volume'] == 0, 1e-6, data['volume'])
    high_volume_efficiency = price_efficiency * returns_per_volume
    
    # Extreme volume acceleration
    volume_at_high_low_ratio = data['volume'] / np.where(data['volume'].rolling(window=5).mean() == 0, 1e-6, data['volume'].rolling(window=5).mean())
    volume_growth = data['volume'].pct_change(periods=3)
    extreme_volume_acceleration = volume_at_high_low_ratio * volume_growth
    
    # 3. Price Compression Expansion with Asymmetry
    # Asymmetric compression
    current_range = data['high'] - data['low']
    prev_range = data['prev_high'] - data['prev_low']
    prev_range = np.where(prev_range == 0, 1e-6, prev_range)
    compression_ratio = current_range / prev_range
    
    # Breakout direction preference
    high_breakout = (data['high'] > data['prev_high']).astype(int)
    low_breakout = (data['low'] < data['prev_low']).astype(int)
    breakout_direction = high_breakout - low_breakout
    
    asymmetric_compression = compression_ratio * breakout_direction
    
    # Biased expansion momentum
    close_open_diff = data['close'] - data['open']
    historical_bias = data['intraday_return'].rolling(window=10, min_periods=7).mean()
    biased_expansion_momentum = close_open_diff * compression_ratio * historical_bias
    
    # 4. Bidirectional Gap Persistence with Anchor Rejection
    # Anchor-rejected gap persistence
    gap_direction = np.sign(data['open'] - data['prev_close'])
    intraday_direction = np.sign(data['close'] - data['open'])
    
    # Price rejection from anchors (distance from high/low to open)
    high_rejection = (data['high'] - data['open']) / np.where(data['open'] == 0, 1e-6, data['open'])
    low_rejection = (data['open'] - data['low']) / np.where(data['open'] == 0, 1e-6, data['open'])
    price_rejection = high_rejection - low_rejection
    
    anchor_rejected_gap = gap_direction * intraday_direction * price_rejection
    
    # Volume-confirmed gap magnitude
    gap_magnitude = np.abs(data['open'] - data['prev_close']) / np.where(np.abs(data['close'] - data['open']) == 0, 1e-6, np.abs(data['close'] - data['open']))
    volume_at_rejection = data['volume'] / np.where(data['volume'].rolling(window=5).mean() == 0, 1e-6, data['volume'].rolling(window=5).mean())
    volume_confirmed_gap = gap_magnitude * volume_at_rejection
    
    # 5. Volatility Clustering Momentum with Efficiency
    # Efficient volatility clustering
    efficiency_divergence = (data['high'] - data['low']) / np.where(data['prev_high'] - data['prev_low'] == 0, 1e-6, data['prev_high'] - data['prev_low'])
    volatility_clustering = data['intraday_return'].abs().rolling(window=5, min_periods=3).std()
    efficient_vol_clustering = efficiency_divergence * volatility_clustering
    
    # Volume-aligned clustered momentum
    volume_trend = data['volume'].rolling(window=5, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_aligned_momentum = data['intraday_return'] * volatility_clustering * volume_trend
    
    # 6. Session Transition Momentum with Clustering
    # Clustered transition momentum
    overnight_intraday_momentum = data['overnight_return'] * data['intraday_return']
    
    # Consecutive directional moves
    positive_moves = (data['intraday_return'] > 0).astype(int)
    consecutive_moves = positive_moves.rolling(window=3).sum()
    clustered_transition = overnight_intraday_momentum * consecutive_moves
    
    # Volume-patterned transition
    volume_alignment = data['volume'] / np.where(data['volume'].rolling(window=10).mean() == 0, 1e-6, data['volume'].rolling(window=10).mean())
    historical_pattern = data['intraday_return'].rolling(window=10, min_periods=7).std()
    volume_patterned_transition = overnight_intraday_momentum * volume_alignment * historical_pattern
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'fragmented_morning_reversal': fragmented_morning_reversal,
        'dispersed_afternoon_reversal': dispersed_afternoon_reversal,
        'high_volume_efficiency': high_volume_efficiency,
        'extreme_volume_acceleration': extreme_volume_acceleration,
        'asymmetric_compression': asymmetric_compression,
        'biased_expansion_momentum': biased_expansion_momentum,
        'anchor_rejected_gap': anchor_rejected_gap,
        'volume_confirmed_gap': volume_confirmed_gap,
        'efficient_vol_clustering': efficient_vol_clustering,
        'volume_aligned_momentum': volume_aligned_momentum,
        'clustered_transition': clustered_transition,
        'volume_patterned_transition': volume_patterned_transition
    })
    
    # Final factor as equal-weighted combination of normalized components
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / np.where(x.rolling(window=20).std() == 0, 1e-6, x.rolling(window=20).std()))
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
