import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factors using multiple intraday and volume-based patterns
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Momentum Divergence
    data['morning_momentum'] = (data['high'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['low']) / data['low']
    
    # Calculate volume ratios
    morning_volume_ratio = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: x.iloc[0] / x.mean() if x.mean() > 0 else 1.0
    )
    afternoon_volume_ratio = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1.0
    )
    
    # Momentum divergence with volume weighting
    momentum_divergence = (
        (data['morning_momentum'] * morning_volume_ratio) - 
        (data['afternoon_momentum'] * afternoon_volume_ratio)
    )
    
    # Factor 2: Volatility-Regime Adaptive Reversal
    volatility = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    high_vol_regime = volatility > volatility.rolling(window=50, min_periods=25).quantile(0.7)
    low_vol_regime = volatility < volatility.rolling(window=50, min_periods=25).quantile(0.3)
    
    # High volatility mean reversion
    short_term_ret = data['close'].pct_change(periods=3)
    volume_spike = data['volume'] > data['volume'].rolling(window=20, min_periods=10).mean() * 1.5
    
    high_vol_signal = -short_term_ret * volume_spike * high_vol_regime
    
    # Low volatility breakout anticipation
    daily_range = (data['high'] - data['low']) / data['close']
    range_contraction = daily_range / daily_range.rolling(window=10, min_periods=5).mean()
    volume_drying = data['volume'] < data['volume'].rolling(window=20, min_periods=10).mean() * 0.8
    
    low_vol_signal = -range_contraction * volume_drying * low_vol_regime
    
    volatility_factor = high_vol_signal.fillna(0) + low_vol_signal.fillna(0)
    
    # Factor 3: Opening Gap Fade Strength
    prev_close = data['close'].shift(1)
    gap_size = (data['open'] - prev_close) / prev_close
    avg_range = (data['high'] - data['low']).rolling(window=10, min_periods=5).mean() / data['close']
    
    relative_gap = gap_size / (avg_range + 1e-8)
    fade_momentum = -np.sign(gap_size) * (data['close'] - data['open']) / data['open']
    
    # Volume confirmation
    fade_volume_intensity = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    gap_factor = relative_gap * fade_momentum * fade_volume_intensity
    
    # Factor 4: Volume-Implied Price Efficiency
    price_movement = (data['close'] - data['open']).abs() / data['open']
    efficiency = price_movement / (data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean() + 1e-8)
    
    # Identify inefficient moves (high volume, low return)
    volume_rank = data['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    return_rank = price_movement.rolling(window=20, min_periods=10).rank(pct=True)
    
    inefficiency_score = volume_rank - return_rank
    efficiency_factor = -inefficiency_score * np.sign(data['close'] - data['open'])
    
    # Factor 5: Closing Range Position Momentum
    close_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Strong close patterns
    strong_high_close = (close_position > 0.7).astype(int) * (data['close'] > data['open']).astype(int)
    strong_low_close = (close_position < 0.3).astype(int) * (data['close'] < data['open']).astype(int)
    
    # Weak close patterns (indecision)
    middle_close = ((close_position >= 0.4) & (close_position <= 0.6)).astype(int)
    
    close_momentum = (
        strong_high_close - strong_low_close - 0.5 * middle_close
    ) * (data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean())
    
    # Factor 6: Intraday Volume Distribution Skew
    # Using rolling window to estimate volume timing patterns
    volume_skew = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.mean() + 1e-8)
    )
    
    # Price response to volume skew
    price_response = data['close'].pct_change() * volume_skew
    volume_skew_factor = -price_response.rolling(window=10, min_periods=5).mean()
    
    # Factor 7: Range Break Failure Detection
    prev_high = data['high'].shift(1)
    prev_low = data['low'].shift(1)
    
    # Failed breakout (tested high but closed low)
    failed_breakout = (
        (data['high'] > prev_high) & 
        (data['close'] < (data['high'] + prev_high) / 2)
    ).astype(int)
    
    # Failed breakdown (tested low but closed high)
    failed_breakdown = (
        (data['low'] < prev_low) & 
        (data['close'] > (data['low'] + prev_low) / 2)
    ).astype(int)
    
    failure_strength = (
        failed_breakdown - failed_breakout
    ) * (data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean())
    
    # Factor 8: Price-Volume Divergence Momentum
    price_trend = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
    )
    
    volume_trend = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.mean() + 1e-8)
    )
    
    # Bullish divergence (price down, volume up)
    bullish_div = ((price_trend < 0) & (volume_trend > 0)).astype(int)
    # Bearish divergence (price up, volume down)
    bearish_div = ((price_trend > 0) & (volume_trend < 0)).astype(int)
    
    divergence_factor = bullish_div - bearish_div
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum_div': momentum_divergence,
        'volatility_adapt': volatility_factor,
        'gap_fade': gap_factor,
        'efficiency': efficiency_factor,
        'close_momentum': close_momentum,
        'volume_skew': volume_skew_factor,
        'failure_detect': failure_strength,
        'divergence': divergence_factor
    })
    
    # Normalize each factor
    normalized_factors = factors.apply(
        lambda x: (x - x.rolling(window=50, min_periods=25).mean()) / 
                 (x.rolling(window=50, min_periods=25).std() + 1e-8)
    )
    
    # Equal weighted combination
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
