import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factors using multiple heuristics including:
    - Intraday Momentum Decay
    - Price Gap Volume Asymmetry  
    - High-Low Compression Breakout
    - Volatility-Regime Adjusted Return
    - Volume-Price Divergence
    - Multi-Timeframe Price Anchoring
    - Opening Auction Imbalance
    - Trend Exhaustion Detection
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 20:
        return factor
    
    # 1. Intraday Momentum Decay Factor
    # Calculate intraday price range
    intraday_range = (data['high'] - data['low']) / data['close']
    
    # Compute intraday return patterns
    intraday_return_open_close = (data['close'] - data['open']) / data['open']
    intraday_return_high_low = (data['high'] - data['low']) / data['low']
    
    # Assess momentum decay using consecutive day patterns
    momentum_decay = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        # Compare current intraday pattern with previous day
        current_pattern = intraday_return_open_close.iloc[i] + intraday_return_high_low.iloc[i]
        prev_pattern = intraday_return_open_close.iloc[i-1] + intraday_return_high_low.iloc[i-1]
        
        # Weight by volume confirmation
        volume_ratio = data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 1e-8)
        momentum_decay.iloc[i] = (current_pattern - prev_pattern) * volume_ratio
    
    # 2. Price Gap Volume Asymmetry
    # Calculate overnight gap
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Measure volume response and acceleration
    volume_acceleration = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Assess asymmetry pattern
    gap_asymmetry = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        current_gap = overnight_gap.iloc[i]
        current_volume_accel = volume_acceleration.iloc[i]
        
        # Different behavior for positive vs negative gaps
        if current_gap > 0:
            # Positive gap: expect volume confirmation
            gap_asymmetry.iloc[i] = current_gap * current_volume_accel
        else:
            # Negative gap: expect volume acceleration
            gap_asymmetry.iloc[i] = current_gap * (2 - current_volume_accel)
    
    # 3. High-Low Compression Breakout Signal
    # Measure trading range compression
    hl_range = (data['high'] - data['low']) / data['close']
    range_ma = hl_range.rolling(window=10, min_periods=5).mean()
    range_compression = hl_range / (range_ma + 1e-8)
    
    # Detect breakout confirmation
    breakout_signal = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        if range_compression.iloc[i] < 0.8:  # Compressed range
            # Check for breakout direction
            price_change = (data['close'].iloc[i] - data['open'].iloc[i]) / data['open'].iloc[i]
            volume_breakout = data['volume'].iloc[i] > data['volume'].rolling(window=5).mean().iloc[i]
            
            if volume_breakout:
                breakout_signal.iloc[i] = price_change * range_compression.iloc[i]
    
    # 4. Volatility-Regime Adjusted Return
    # Characterize volatility environment
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20, min_periods=10).std()
    vol_regime = volatility > volatility.rolling(window=50, min_periods=25).mean()
    
    # Adjust return expectations based on regime
    regime_adjusted_return = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        current_return = returns.iloc[i]
        if vol_regime.iloc[i]:
            # High volatility regime: dampen returns
            regime_adjusted_return.iloc[i] = current_return * 0.7
        else:
            # Low volatility regime: amplify returns
            regime_adjusted_return.iloc[i] = current_return * 1.3
    
    # 5. Volume-Price Divergence Oscillator
    # Track volume trends
    volume_ma = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_momentum = data['volume'] / volume_ma
    
    # Compare price action for divergence
    price_momentum = data['close'] / data['close'].rolling(window=10, min_periods=5).mean()
    
    # Detect divergence signals
    divergence = volume_momentum - price_momentum
    
    # 6. Multi-Timeframe Price Anchoring
    # Establish price reference points
    short_term_anchor = data['close'].rolling(window=5, min_periods=3).mean()
    medium_term_anchor = data['close'].rolling(window=20, min_periods=10).mean()
    
    # Volume-weighted price anchor
    vwap = (data['close'] * data['volume']).rolling(window=5, min_periods=3).sum() / \
           data['volume'].rolling(window=5, min_periods=3).sum()
    
    # Measure price positioning relative to anchors
    price_position = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        current_price = data['close'].iloc[i]
        
        # Distance from key anchors
        dist_short = (current_price - short_term_anchor.iloc[i]) / short_term_anchor.iloc[i]
        dist_medium = (current_price - medium_term_anchor.iloc[i]) / medium_term_anchor.iloc[i]
        dist_vwap = (current_price - vwap.iloc[i]) / vwap.iloc[i]
        
        # Combined positioning score
        price_position.iloc[i] = (dist_short + dist_medium + dist_vwap) / 3
    
    # 7. Opening Auction Imbalance Factor
    opening_imbalance = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        # Price gap fill rate
        gap = (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        
        # Volume distribution pattern (first hour proxy)
        if i < len(data) - 1:
            # Use current day's volume as proxy for opening period
            daily_volume_ratio = data['volume'].iloc[i] / data['volume'].rolling(window=5).mean().iloc[i]
            
            # Imbalance strength
            opening_imbalance.iloc[i] = gap * daily_volume_ratio
    
    # 8. Trend Exhaustion Detection
    # Identify overextended moves
    price_trend = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) == 10 else np.nan
    )
    
    # Volume exhaustion signals
    volume_trend = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) == 10 else np.nan
    )
    
    # Assess reversal probability
    trend_exhaustion = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        if abs(price_trend.iloc[i]) > 0.05:  # Significant move
            # Check for volume exhaustion (divergence)
            if (price_trend.iloc[i] > 0 and volume_trend.iloc[i] < 0) or \
               (price_trend.iloc[i] < 0 and volume_trend.iloc[i] > 0):
                trend_exhaustion.iloc[i] = -price_trend.iloc[i]  # Expect reversal
    
    # Combine all factors with equal weighting
    factors = [
        momentum_decay.fillna(0),
        gap_asymmetry.fillna(0),
        breakout_signal.fillna(0),
        regime_adjusted_return.fillna(0),
        divergence.fillna(0),
        price_position.fillna(0),
        opening_imbalance.fillna(0),
        trend_exhaustion.fillna(0)
    ]
    
    # Normalize each factor
    normalized_factors = []
    for f in factors:
        if len(f.dropna()) > 0:
            f_normalized = (f - f.mean()) / (f.std() + 1e-8)
            normalized_factors.append(f_normalized)
    
    # Combine normalized factors
    if normalized_factors:
        combined_factor = sum(normalized_factors) / len(normalized_factors)
        factor = combined_factor
    
    return factor
