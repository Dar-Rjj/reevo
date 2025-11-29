import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Range Efficiency with Volume Confirmation
    # Calculate Intraday Range Efficiency
    morning_momentum = (df['high'] - df['open']) + (df['close'] - df['low'])
    price_range = df['high'] - df['low']
    range_efficiency = morning_momentum / price_range.replace(0, np.nan)
    
    # Detect Volume Confirmation
    # Assuming first hour volume is not available, use morning volume proxy
    morning_volume_ratio = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.iloc[0] / x.mean() if x.mean() > 0 else np.nan)
    volume_divergence = df['volume'].rolling(window=10, min_periods=1).apply(lambda x: (x.iloc[:5].mean() - x.iloc[5:].mean()) / x.mean() if x.mean() > 0 else np.nan)
    volume_confirmation = morning_volume_ratio * volume_divergence
    
    # Combine Efficiency and Volume Signals
    close_position = (df['close'] - df['low']) / price_range.replace(0, np.nan)
    combined_signal = range_efficiency * volume_confirmation * close_position
    
    # Gap Fill Momentum with Range Asymmetry
    # Analyze Opening Gap Behavior
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
    gap_fill = (df['close'] - df['open']) / (df['close'].shift(1) - df['open']).replace(0, np.nan)
    gap_fill_momentum = overnight_gap * gap_fill
    
    # Compute Range Asymmetry
    upper_extension = df['high'] - df['open']
    lower_extension = df['open'] - df['low']
    asymmetry_ratio = np.log(upper_extension / lower_extension.replace(0, np.nan))
    large_trade_impact = df['amount'] / df['amount'].rolling(window=20, min_periods=1).mean()
    gap_asymmetry_signal = gap_fill_momentum * asymmetry_ratio * large_trade_impact
    
    # Volume-Weighted Price Acceleration
    # Calculate Multi-scale Price Acceleration
    short_term_momentum = df['close'] - df['close'].shift(1)
    acceleration = short_term_momentum - short_term_momentum.shift(1)
    pattern_persistence = (acceleration > 0).rolling(window=3, min_periods=1).sum()
    
    # Apply Volume Significance Weighting
    volume_profile = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    volume_weighted_signal = acceleration * volume_profile
    
    # Range breakout confirmation
    range_ratio = price_range / price_range.rolling(window=20, min_periods=1).mean()
    breakout_strength = (df['close'] > df['high'].shift(1)).astype(int) * range_ratio
    acceleration_signal = volume_weighted_signal * pattern_persistence * breakout_strength
    
    # Opening Auction Efficiency with Position Persistence
    opening_efficiency = (df['open'] - df['close'].shift(1)) / price_range.replace(0, np.nan)
    auction_strength = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    
    position_strength = close_position.rolling(window=5, min_periods=1).mean()
    position_persistence = (close_position > 0.5).rolling(window=3, min_periods=1).sum()
    
    amount_momentum = df['amount'] * short_term_momentum / df['amount'].rolling(window=20, min_periods=1).mean()
    auction_signal = opening_efficiency * auction_strength * position_strength * position_persistence * amount_momentum
    
    # Range Breakout Efficiency with Volume Divergence
    range_contraction = price_range / price_range.rolling(window=20, min_periods=1).mean()
    breakout_efficiency = (df['close'] > df['high'].shift(1)).astype(int) * short_term_momentum
    
    volume_distribution = df['volume'].rolling(window=10, min_periods=1).apply(lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else np.nan)
    volume_divergence_ratio = volume_distribution / volume_distribution.rolling(window=20, min_periods=1).mean()
    
    # Support/resistance levels using recent highs/lows
    resistance_level = df['high'].rolling(window=10, min_periods=1).max()
    support_level = df['low'].rolling(window=10, min_periods=1).min()
    level_strength = (df['close'] - support_level) / (resistance_level - support_level).replace(0, np.nan)
    
    breakout_signal = breakout_efficiency * volume_divergence_ratio * range_contraction * level_strength
    
    # Combine all signals with equal weighting
    final_factor = (combined_signal + gap_asymmetry_signal + acceleration_signal + auction_signal + breakout_signal) / 5
    
    return final_factor
