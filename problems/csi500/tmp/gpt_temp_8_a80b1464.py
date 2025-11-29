import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Components
    # Short-term intraday trend: (Close - Open) / Open
    short_term_trend = (df['close'] - df['open']) / df['open']
    
    # Medium-term momentum: Close / Close 5 days ago - 1
    medium_term_momentum = df['close'] / df['close'].shift(5) - 1
    
    # Volatility adjustment: (High - Low) / Open
    volatility_adjustment = (df['high'] - df['low']) / df['open']
    
    # Momentum acceleration: (Short-term trend - Medium-term momentum) / Volatility adjustment
    # Avoid division by zero
    volatility_adjustment_safe = volatility_adjustment.replace(0, np.nan)
    momentum_acceleration = (short_term_trend - medium_term_momentum) / volatility_adjustment_safe
    
    # Volume-Price Confirmation
    # Volume ratio: Current volume / 20-day average volume
    volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Volume-price correlation: 10-day correlation(volume, price changes)
    price_changes = df['close'].pct_change()
    volume_price_corr = df['volume'].rolling(window=10, min_periods=2).corr(price_changes)
    
    # Volume momentum: Current volume / Volume 5 days ago - 1
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume breakout strength: Volume ratio × absolute(price change) × Volume momentum
    volume_breakout_strength = volume_ratio * abs(price_changes) * volume_momentum
    
    # Factor Construction
    # Raw factor: Momentum acceleration × Volume breakout strength
    raw_factor = momentum_acceleration * volume_breakout_strength
    
    # Trend persistence filter: Sign consistency count over 3 days
    sign_consistency = raw_factor.rolling(window=3, min_periods=1).apply(
        lambda x: sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 0 else 1
    )
    
    # Correlation confirmation: Multiply by Volume-price correlation sign
    correlation_confirmation = np.sign(volume_price_corr)
    
    # Final factor: Raw factor × Trend persistence filter × Correlation confirmation
    final_factor = raw_factor * sign_consistency * correlation_confirmation
    
    return final_factor
