import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # High-Low Volatility Regime
    # Range Expansion Detection
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['range_ma5'] = data['high_low_range'].rolling(window=5).mean()
    data['range_ma20'] = data['high_low_range'].rolling(window=20).mean()
    
    # High-Low Range Acceleration
    data['range_acceleration'] = data['range_ma5'] / data['range_ma20'] - 1
    
    # Range vs Historical Comparison
    data['range_percentile'] = data['high_low_range'].rolling(window=60).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Volume Confirmation
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma20'] = data['volume'].rolling(window=20).mean()
    data['volume_regime'] = data['volume_ma5'] / data['volume_ma20'] - 1
    
    # Volume-Range Coherence
    data['volume_range_coherence'] = (data['range_acceleration'] * data['volume_regime'])
    
    # Opening Gap Momentum
    # Gap Magnitude Analysis
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_ma10'] = data['opening_gap'].rolling(window=10).mean()
    data['gap_std10'] = data['opening_gap'].rolling(window=10).std()
    
    # Current Gap vs Historical
    data['gap_zscore'] = (data['opening_gap'] - data['gap_ma10']) / data['gap_std10'].replace(0, 1)
    
    # Gap Persistence Pattern
    data['gap_persistence'] = data['opening_gap'].rolling(window=3).apply(
        lambda x: 1 if all(x > 0) else (-1 if all(x < 0) else 0)
    )
    
    # Intraday Momentum
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['gap_close_relationship'] = data['opening_gap'] * data['intraday_return']
    
    # Volume-Gap Divergence
    data['volume_gap_divergence'] = data['opening_gap'] * data['volume_regime']
    
    # Intraday Liquidity Reversal
    # Position-Based Signals
    data['opening_strength'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, 1)
    data['closing_weakness'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, 1)
    
    # Liquidity Confirmation
    data['amount_ma5'] = data['amount'].rolling(window=5).mean()
    data['amount_regime'] = data['amount'] / data['amount_ma5'] - 1
    data['volume_acceleration'] = data['volume'] / data['volume_ma5'] - 1
    
    # Amount-Driven Reversal
    data['amount_reversal'] = data['amount_regime'] * data['intraday_return']
    
    # Price-Volume Dynamics
    # Acceleration Patterns
    data['price_momentum'] = data['close'].pct_change(periods=5)
    data['volume_momentum'] = data['volume'].pct_change(periods=5)
    
    # Price Momentum Change
    data['price_acceleration'] = data['price_momentum'] - data['price_momentum'].shift(1)
    
    # Volume Momentum Change
    data['volume_momentum_change'] = data['volume_momentum'] - data['volume_momentum'].shift(1)
    
    # Coherence Assessment
    data['direction_alignment'] = np.sign(data['price_momentum']) * np.sign(data['volume_momentum'])
    data['market_phase'] = data['price_momentum'].rolling(window=10).apply(
        lambda x: 1 if x.mean() > 0 else (-1 if x.mean() < 0 else 0)
    )
    
    # Liquidity Breakout
    # Momentum Convergence
    data['price_breakout'] = (data['close'] - data['close'].rolling(window=20).mean()) / data['close'].rolling(window=20).std().replace(0, 1)
    data['liquidity_surge'] = data['volume'] / data['volume_ma20'] - 1
    
    # Pattern Validation
    data['range_break'] = data['high_low_range'] / data['range_ma20'] - 1
    data['volume_expansion'] = data['volume'] / data['volume_ma20'] - 1
    
    # Combine factors with weights
    factor = (
        0.15 * data['range_acceleration'] +
        0.12 * data['range_percentile'] +
        0.10 * data['volume_range_coherence'] +
        0.08 * data['gap_zscore'] +
        0.07 * data['gap_persistence'] +
        0.09 * data['gap_close_relationship'] +
        0.06 * data['volume_gap_divergence'] +
        0.08 * data['opening_strength'] +
        0.07 * data['closing_weakness'] +
        0.05 * data['amount_reversal'] +
        0.04 * data['price_acceleration'] +
        0.03 * data['volume_momentum_change'] +
        0.03 * data['direction_alignment'] +
        0.02 * data['price_breakout'] +
        0.01 * data['liquidity_surge']
    )
    
    # Clean up and return
    result = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    return result
