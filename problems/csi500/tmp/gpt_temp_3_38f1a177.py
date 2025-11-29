import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate composite alpha factor using multiple intraday and daily patterns
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Momentum Exhaustion Divergence
    # Calculate intraday momentum strength
    high_momentum = (data['high'] - data['open']) / data['open']
    low_momentum = (data['open'] - data['low']) / data['open']
    
    # Morning strength (first 30 minutes proxy using open-to-high)
    morning_strength = high_momentum.rolling(window=5, min_periods=3).mean()
    
    # Afternoon weakness (using close relative to high)
    afternoon_weakness = ((data['high'] - data['close']) / data['high']).rolling(window=5, min_periods=3).mean()
    
    # Volume-confirmed divergence
    volume_trend = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan
    )
    momentum_divergence = (morning_strength - afternoon_weakness) * np.sign(volume_trend)
    
    # 2. Price-Volume Fractal Efficiency
    # Multi-scale price patterns
    micro_range = (data['high'] - data['low']).rolling(window=3, min_periods=2).mean()
    macro_range = (data['high'] - data['low']).rolling(window=10, min_periods=7).mean()
    
    # Price movement efficiency
    price_efficiency = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    price_efficiency = price_efficiency.rolling(window=5, min_periods=3).mean()
    
    # Volume distribution analysis
    volume_std = data['volume'].rolling(window=10, min_periods=5).std()
    volume_mean = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_efficiency = volume_std / volume_mean.replace(0, np.nan)
    
    fractal_efficiency = (micro_range / macro_range.replace(0, np.nan)) * price_efficiency * volume_efficiency
    
    # 3. Opening Auction Imbalance Persistence
    # Opening characteristics
    gap_size = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    opening_volume_ratio = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Gap fill vs expansion
    gap_fill_efficiency = ((data['high'] - data['low']) / gap_size.abs().replace(0, np.nan)).rolling(window=5, min_periods=3).mean()
    
    # Volume decay patterns
    volume_decay = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.exp(-0.5 * np.polyfit(range(len(x)), x, 1)[0]) if len(x) >= 3 and np.polyfit(range(len(x)), x, 1)[0] < 0 else 1.0
    )
    
    auction_imbalance = gap_size * opening_volume_ratio * gap_fill_efficiency * volume_decay
    
    # 4. Multi-timeframe Range Compression-Expansion
    # Range compression detection
    short_term_range = (data['high'] - data['low']).rolling(window=5, min_periods=3).std()
    medium_term_range = (data['high'] - data['low']).rolling(window=20, min_periods=10).std()
    range_compression = short_term_range / medium_term_range.replace(0, np.nan)
    
    # Volume breakout confirmation
    volume_breakout = (data['volume'] > data['volume'].rolling(window=20, min_periods=10).quantile(0.8)).astype(float)
    price_breakout = ((data['high'] - data['low']) > (data['high'] - data['low']).rolling(window=20, min_periods=10).quantile(0.8)).astype(float)
    
    range_expansion = range_compression * volume_breakout * price_breakout
    
    # 5. Close-Relative Position Hierarchy
    # Daily position metrics
    close_to_high = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    close_to_low = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Position persistence
    strong_closes = (close_to_low > 0.7).rolling(window=3, min_periods=2).sum()
    position_reversal = (close_to_high.diff(1) > 0.1).rolling(window=5, min_periods=3).sum()
    
    position_hierarchy = (close_to_low - close_to_high) * strong_closes * (1 - position_reversal / 5)
    
    # 6. Volume-Weighted Price Acceleration
    # Price acceleration (second derivative)
    price_velocity = data['close'].diff(1)
    price_acceleration = price_velocity.diff(1)
    
    # Multi-period acceleration
    multi_accel = price_acceleration.rolling(window=5, min_periods=3).mean()
    
    # Volume-weighted acceleration
    volume_weight = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    volume_acceleration = multi_accel * volume_weight
    
    # 7. Intraday Support-Resistance Efficiency
    # Key intraday levels
    morning_high = data['high'].rolling(window=5, min_periods=3).apply(lambda x: x[:3].max() if len(x) >= 3 else np.nan)
    morning_low = data['low'].rolling(window=5, min_periods=3).apply(lambda x: x[:3].min() if len(x) >= 3 else np.nan)
    
    # Bounce efficiency (price reaction near levels)
    high_bounce = ((morning_high - data['close']) / (data['high'] - data['low']).replace(0, np.nan)).abs()
    low_bounce = ((data['close'] - morning_low) / (data['high'] - data['low']).replace(0, np.nan)).abs()
    
    support_resistance_efficiency = (high_bounce + low_bounce) / 2
    
    # 8. Amount-Based Momentum Confirmation
    # Large transaction impact
    large_amount_threshold = data['amount'].rolling(window=20, min_periods=10).quantile(0.8)
    large_trades = (data['amount'] > large_amount_threshold).astype(float)
    
    # Amount-momentum alignment
    price_momentum = data['close'].pct_change(1)
    amount_momentum_alignment = large_trades * np.sign(price_momentum) * np.abs(price_momentum)
    
    amount_confirmation = amount_momentum_alignment.rolling(window=5, min_periods=3).mean()
    
    # 9. Price-Range Asymmetry Detection
    # Range asymmetry
    upward_extension = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    downward_extension = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    range_asymmetry = upward_extension - downward_extension
    
    # Asymmetry patterns
    consistent_asymmetry = range_asymmetry.rolling(window=5, min_periods=3).std()
    asymmetry_reversal = (range_asymmetry.diff(1).abs() > 0.2).rolling(window=5, min_periods=3).sum()
    
    asymmetry_detection = range_asymmetry * (1 - consistent_asymmetry) * (1 - asymmetry_reversal / 5)
    
    # 10. Volume-Profile Shape Analysis
    # Volume by price level (simplified)
    high_volume_zones = (data['close'] > data['open']).astype(float) * data['volume']
    low_volume_zones = (data['close'] < data['open']).astype(float) * data['volume']
    
    volume_profile_ratio = (high_volume_zones - low_volume_zones) / (high_volume_zones + low_volume_zones).replace(0, np.nan)
    
    # Profile shape changes
    profile_development = volume_profile_ratio.diff(1).abs().rolling(window=5, min_periods=3).mean()
    profile_shift = (volume_profile_ratio.rolling(window=5, min_periods=3).mean().diff(1) > 0.1).astype(float)
    
    volume_profile_analysis = volume_profile_ratio * (1 - profile_development) * profile_shift
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum_divergence': momentum_divergence,
        'fractal_efficiency': fractal_efficiency,
        'auction_imbalance': auction_imbalance,
        'range_expansion': range_expansion,
        'position_hierarchy': position_hierarchy,
        'volume_acceleration': volume_acceleration,
        'support_resistance': support_resistance_efficiency,
        'amount_confirmation': amount_confirmation,
        'asymmetry_detection': asymmetry_detection,
        'volume_profile': volume_profile_analysis
    })
    
    # Normalize each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=50, min_periods=30).mean()) / 
                                     x.rolling(window=50, min_periods=30).std().replace(0, np.nan))
    
    # Equal-weighted composite factor
    composite_factor = normalized_factors.mean(axis=1)
    
    return composite_factor
