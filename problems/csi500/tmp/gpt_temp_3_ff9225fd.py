import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factors using various market microstructure heuristics
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Volume Adjusted High-Low Range Momentum
    # Calculate High-Low Range
    hl_range = data['high'] - data['low']
    
    # Calculate Volume Ratio (current volume / previous volume)
    volume_ratio = data['volume'] / data['volume'].shift(1)
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Adjust range by volume ratio
    adjusted_range = hl_range * volume_ratio
    
    # Compute momentum (current adjusted range / previous adjusted range - 1)
    range_momentum = adjusted_range / adjusted_range.shift(1) - 1
    range_momentum = range_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Factor 2: Price Gap Amplification Factor
    # Calculate overnight gap
    overnight_gap = (data['open'] / data['close'].shift(1) - 1).abs()
    
    # Set gap threshold (95th percentile of historical gaps)
    gap_threshold = overnight_gap.rolling(window=20, min_periods=10).quantile(0.95)
    
    # Identify significant gaps
    significant_gaps = overnight_gap > gap_threshold
    
    # Track gap direction and count consecutive days
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    gap_persistence = significant_gaps.astype(int) * gap_direction
    
    # Count consecutive days with same gap direction
    consecutive_count = gap_persistence.groupby(
        (gap_persistence != gap_persistence.shift(1)).cumsum()
    ).cumcount() + 1
    consecutive_count = consecutive_count * gap_persistence.abs()
    
    # Volume trend (5-day moving average)
    volume_trend = data['volume'].rolling(window=5, min_periods=3).mean() / \
                   data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Combine gap factors
    gap_factor = overnight_gap * consecutive_count * volume_trend
    
    # Factor 3: Volatility Regime Switch Detector
    # Calculate rolling volatility using high-low range
    volatility = hl_range.rolling(window=10, min_periods=5).std()
    historical_volatility = hl_range.rolling(window=30, min_periods=15).std()
    
    # Identify regime changes
    vol_ratio = volatility / historical_volatility
    regime_change_up = (vol_ratio > 1.5) & (vol_ratio.shift(1) <= 1.5)
    regime_change_down = (vol_ratio < 0.7) & (vol_ratio.shift(1) >= 0.7)
    
    # Generate volatility signal
    vol_signal = pd.Series(0, index=data.index)
    vol_signal[regime_change_up] = 1
    vol_signal[regime_change_down] = -1
    
    # Factor 4: Liquidity-Efficient Price Movement
    # Calculate absolute return
    abs_return = (data['close'] / data['close'].shift(1) - 1).abs()
    
    # Calculate price efficiency (return per unit volume)
    price_efficiency = abs_return / (data['volume'] + 1e-10)
    
    # Compare small vs large volume moves
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Identify volume-price mismatches
    efficiency_percentile = price_efficiency.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Generate liquidity signal
    liquidity_signal = efficiency_percentile - volume_percentile
    
    # Factor 5: Intraday Momentum Persistence
    # Calculate morning session momentum (open to high/low)
    morning_high_momentum = (data['high'] - data['open']) / data['open']
    morning_low_momentum = (data['open'] - data['low']) / data['open']
    
    # Determine morning trend (positive if high momentum > low momentum)
    morning_trend = np.sign(morning_high_momentum - morning_low_momentum)
    
    # Calculate afternoon session (assuming midday is average of open and close)
    midday_price = (data['open'] + data['close']) / 2
    afternoon_momentum = (data['close'] - midday_price) / midday_price
    
    # Compare afternoon with morning trend
    intraday_signal = morning_trend * afternoon_momentum
    
    # Factor 6: Price Compression Expansion Cycle
    # Calculate range percentile
    range_percentile = hl_range.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Identify compression (narrow range) and expansion (wide range)
    compression = range_percentile < 0.2
    expansion = range_percentile > 0.8
    
    # Measure expansion magnitude
    expansion_magnitude = hl_range / hl_range.rolling(window=5, min_periods=3).mean()
    
    # Generate compression-expansion signal
    compression_signal = pd.Series(0, index=data.index)
    compression_signal[compression] = -expansion_magnitude[compression]
    compression_signal[expansion] = expansion_magnitude[expansion]
    
    # Factor 7: Volume-Price Divergence Oscillator
    # Calculate price momentum (3-day return)
    price_momentum = data['close'].pct_change(3)
    
    # Calculate volume momentum (3-day volume change)
    volume_momentum = data['volume'].pct_change(3)
    
    # Generate divergence signal
    divergence_signal = price_momentum - volume_momentum
    
    # Factor 8: Opening Auction Strength Indicator
    # Since we don't have intraday 30-minute data, approximate with daily range
    opening_strength = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-10)
    opening_strength = opening_strength.replace([np.inf, -np.inf], 0)
    
    # Calculate opening volume ratio (first 30 minutes approximated by daily pattern)
    # Use the fact that strong openings often have high early volume
    volume_concentration = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Generate opening signal
    opening_signal = opening_strength * volume_concentration
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'range_momentum': range_momentum,
        'gap_factor': gap_factor,
        'vol_signal': vol_signal,
        'liquidity_signal': liquidity_signal,
        'intraday_signal': intraday_signal,
        'compression_signal': compression_signal,
        'divergence_signal': divergence_signal,
        'opening_signal': opening_signal
    })
    
    # Normalize each factor
    normalized_factors = factors.apply(
        lambda x: (x - x.rolling(window=20, min_periods=10).mean()) / 
                  (x.rolling(window=20, min_periods=10).std() + 1e-10)
    )
    
    # Final combined factor (equal weighted)
    combined_factor = normalized_factors.mean(axis=1)
    
    return combined_factor
