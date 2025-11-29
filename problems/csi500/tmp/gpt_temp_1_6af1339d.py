import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday microstructure-based alpha factor combining:
    - Intraday momentum persistence
    - Opening gap microstructure
    - Liquidity clustering patterns
    - Range compression dynamics
    - Amount distribution analysis
    - Closing auction effects
    """
    data = df.copy()
    
    # Feature 1: Consecutive intraday direction persistence
    # Calculate intraday price movement direction (1 for up, -1 for down, 0 for flat)
    intraday_return = (data['close'] - data['open']) / data['open']
    direction = np.sign(intraday_return)
    
    # Rolling persistence of intraday direction (5-day window)
    direction_persistence = direction.rolling(window=5).apply(
        lambda x: np.sum(x[:-1] == x[-1]) if len(x) == 5 else np.nan
    )
    
    # Feature 2: Volume-adjusted price movement efficiency
    price_range = (data['high'] - data['low']) / data['open']
    volume_efficiency = np.abs(intraday_return) / (data['volume'] / data['volume'].rolling(window=20).mean())
    volume_efficiency = volume_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Feature 3: Overnight information absorption
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_absorption = np.abs(overnight_gap) / (price_range + 1e-8)
    gap_absorption = gap_absorption.replace([np.inf, -np.inf], np.nan)
    
    # Feature 4: Opening liquidity concentration
    opening_volume_ratio = data['volume'] / data['volume'].rolling(window=30).mean()
    opening_liquidity = opening_volume_ratio * np.abs(overnight_gap)
    
    # Feature 5: Transaction size distribution patterns
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    trade_size_std = avg_trade_size.rolling(window=10).std()
    liquidity_clustering = avg_trade_size / (trade_size_std + 1e-8)
    
    # Feature 6: Amount-based price impact
    price_impact = np.abs(intraday_return) / (data['amount'] / data['amount'].rolling(window=20).mean())
    price_impact = price_impact.replace([np.inf, -np.inf], np.nan)
    
    # Feature 7: Intraday volatility contraction
    daily_range = (data['high'] - data['low']) / data['open']
    range_ma_5 = daily_range.rolling(window=5).mean()
    range_ma_20 = daily_range.rolling(window=20).mean()
    range_compression = (range_ma_5 - range_ma_20) / (range_ma_20 + 1e-8)
    
    # Feature 8: Breakout initiation strength
    close_to_high = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    breakout_strength = close_to_high * np.sqrt(data['volume'] / data['volume'].rolling(window=20).mean())
    
    # Feature 9: Transaction size concentration
    amount_ma_5 = data['amount'].rolling(window=5).mean()
    amount_ma_20 = data['amount'].rolling(window=20).mean()
    amount_concentration = (amount_ma_5 - amount_ma_20) / (amount_ma_20 + 1e-8)
    
    # Feature 10: Amount-weighted price reversals
    prev_close = data['close'].shift(1)
    price_reversal = -np.sign(intraday_return) * (data['amount'] / data['amount'].rolling(window=20).mean())
    
    # Feature 11: End-of-day liquidity patterns
    closing_volume_ratio = data['volume'] / data['volume'].rolling(window=30).mean()
    eod_liquidity = closing_volume_ratio * np.abs(intraday_return)
    
    # Feature 12: Overnight gap precursors
    prev_intraday_range = ((data['high'].shift(1) - data['low'].shift(1)) / data['open'].shift(1))
    gap_precursor = overnight_gap / (prev_intraday_range + 1e-8)
    gap_precursor = gap_precursor.replace([np.inf, -np.inf], np.nan)
    
    # Combine features with appropriate weights
    factor = (
        0.15 * direction_persistence.fillna(0) +
        0.12 * volume_efficiency.fillna(0) +
        0.10 * gap_absorption.fillna(0) +
        0.08 * opening_liquidity.fillna(0) +
        0.09 * liquidity_clustering.fillna(0) +
        0.10 * price_impact.fillna(0) +
        0.08 * range_compression.fillna(0) +
        0.09 * breakout_strength.fillna(0) +
        0.07 * amount_concentration.fillna(0) +
        0.06 * price_reversal.fillna(0) +
        0.04 * eod_liquidity.fillna(0) +
        0.02 * gap_precursor.fillna(0)
    )
    
    # Final normalization
    factor = (factor - factor.rolling(window=60).mean()) / (factor.rolling(window=60).std() + 1e-8)
    
    return factor
