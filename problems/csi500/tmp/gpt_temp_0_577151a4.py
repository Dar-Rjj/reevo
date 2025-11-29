import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Intraday Persistence factor
    """
    data = df.copy()
    
    # Calculate daily returns and intraday metrics
    data['daily_return'] = data['close'].pct_change()
    data['intraday_high_low'] = (data['high'] - data['low']) / data['open']
    data['close_to_open'] = (data['close'] - data['open']) / data['open']
    
    # Intraday Price Persistence Component
    # Consecutive same-direction intraday moves
    data['intraday_direction'] = np.sign(data['close_to_open'])
    data['consecutive_direction'] = data['intraday_direction'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    ).reset_index(level=0, drop=True)
    
    # Magnitude of persistent moves
    data['persistence_magnitude'] = data['consecutive_direction'] * abs(data['close_to_open'])
    
    # Volume Divergence Component
    # Volume trend vs price trend
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['price_ma_5'] = data['close'].rolling(window=5, min_periods=3).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma_5'] - 1
    data['price_trend'] = data['close'] / data['price_ma_5'] - 1
    data['volume_price_divergence'] = data['volume_trend'] - data['price_trend']
    
    # Volume acceleration during persistence
    data['volume_acceleration'] = data['volume'].pct_change() * data['consecutive_direction']
    
    # Opening Gap Analysis
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_fill_ratio'] = (data['close'] - data['open']) / (data['prev_close'] - data['open'])
    data['gap_persistence'] = np.where(
        np.sign(data['opening_gap']) == np.sign(data['close_to_open']),
        abs(data['opening_gap']), 0
    )
    
    # Multi-timeframe Confirmation
    # Short-term persistence (1-3 days)
    data['short_term_persistence'] = data['close_to_open'].rolling(window=3, min_periods=2).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(1, len(x)-1),
        raw=False
    )
    
    # Medium-term divergence patterns (5-10 days)
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=7).mean()
    data['price_ma_10'] = data['close'].rolling(window=10, min_periods=7).mean()
    data['medium_term_divergence'] = (
        (data['volume'] / data['volume_ma_10'] - 1) - 
        (data['close'] / data['price_ma_10'] - 1)
    )
    
    # Long-term volume accumulation (20+ days)
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=15).mean()
    data['long_term_accumulation'] = data['volume'] / data['volume_ma_20'] - 1
    
    # Combine components with weights
    factor = (
        0.25 * data['persistence_magnitude'] +
        0.20 * data['volume_price_divergence'] +
        0.15 * data['volume_acceleration'] +
        0.15 * data['gap_persistence'] +
        0.10 * data['short_term_persistence'] +
        0.10 * data['medium_term_divergence'] +
        0.05 * data['long_term_accumulation']
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60, min_periods=30).mean()) / factor.rolling(window=60, min_periods=30).std()
    
    return factor
