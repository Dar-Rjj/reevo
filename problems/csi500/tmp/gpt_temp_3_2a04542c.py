import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Alpha Factor combining intraday momentum decay, range compression breakout,
    volatility-regime reversal, opening imbalance with trend exhaustion, and price anchoring.
    """
    data = df.copy()
    
    # Intraday Momentum Decay with Volume Asymmetry
    # Morning Session Momentum (Open to Midday approximation)
    data['morning_return'] = (data['high'].rolling(window=5).mean() - data['open']) / data['open']
    data['morning_volume_intensity'] = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean()
    
    # Afternoon Session Momentum (Midday to Close approximation)
    data['afternoon_return'] = (data['close'] - data['low'].rolling(window=5).mean()) / data['low'].rolling(window=5).mean()
    
    # Full Day Momentum
    data['daily_return'] = (data['close'] - data['open']) / data['open']
    
    # Momentum Decay Patterns
    data['momentum_persistence'] = data['daily_return'].rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    data['morning_afternoon_carryover'] = data['morning_return'] - data['afternoon_return'].shift(1)
    data['session_momentum_change'] = (data['morning_return'] - data['afternoon_return']).abs()
    
    # Volume-Price Asymmetry
    data['volume_momentum_alignment'] = data['morning_volume_intensity'] * np.sign(data['morning_return'])
    data['gap_effect'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_momentum_interaction'] = data['gap_effect'] * data['morning_return']
    
    # Range Compression Breakout with Volume Acceleration
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['multi_day_range'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['range_ratio'] = data['daily_range'] / (data['multi_day_range'] / data['close'].rolling(window=5).mean())
    
    data['range_narrowing'] = data['daily_range'].rolling(window=5).apply(
        lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
    )
    
    # Breakout Detection
    data['breakout_strength'] = (data['close'] - data['high'].rolling(window=5).max()) / data['close'].rolling(window=5).std()
    data['volume_acceleration'] = data['volume'] / data['volume'].rolling(window=10).mean() - 1
    
    # Volatility-Regime Momentum Reversal
    data['volatility_hl'] = (data['high'] - data['low']).rolling(window=10).std() / data['close'].rolling(window=10).mean()
    data['volatility_cc'] = data['close'].pct_change().rolling(window=10).std()
    
    volatility_regime = (data['volatility_hl'] > data['volatility_hl'].rolling(window=20).quantile(0.7)).astype(int)
    data['regime_specific_momentum'] = data['daily_return'] * (1 - 2 * volatility_regime)
    
    # Opening Imbalance with Trend Exhaustion
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fill_rate'] = (data['close'] - data['open']) / data['overnight_gap'].abs().replace(0, np.nan)
    
    data['momentum_streak'] = (data['daily_return'] > 0).rolling(window=5).sum() - (data['daily_return'] < 0).rolling(window=5).sum()
    data['volume_exhaustion'] = (data['volume'] < data['volume'].rolling(window=10).quantile(0.3)).astype(int)
    
    # Price Anchoring with Volume-Price Divergence
    data['vwap'] = (data['close'] * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    data['anchor_distance'] = (data['close'] - data['vwap']) / data['close'].rolling(window=10).std()
    
    data['volume_cluster'] = (data['volume'] > data['volume'].rolling(window=20).quantile(0.8)).astype(int)
    data['price_volume_divergence'] = data['anchor_distance'] * (1 - 2 * data['volume_cluster'])
    
    # Composite Factor Calculation
    # Intraday Momentum Decay Component
    momentum_decay = (
        0.3 * data['momentum_persistence'] +
        0.25 * data['morning_afternoon_carryover'] +
        0.2 * data['volume_momentum_alignment'] +
        0.25 * data['gap_momentum_interaction']
    )
    
    # Range Breakout Component
    range_breakout = (
        0.4 * data['breakout_strength'] +
        0.35 * data['volume_acceleration'] +
        0.25 * data['range_narrowing']
    )
    
    # Volatility Regime Component
    volatility_factor = (
        0.5 * data['regime_specific_momentum'] +
        0.3 * data['volatility_hl'] +
        0.2 * data['volatility_cc']
    )
    
    # Opening Imbalance Component
    opening_imbalance = (
        0.35 * data['overnight_gap'] +
        0.3 * data['gap_fill_rate'] +
        0.2 * data['momentum_streak'] +
        0.15 * data['volume_exhaustion']
    )
    
    # Price Anchoring Component
    price_anchoring = (
        0.4 * data['anchor_distance'] +
        0.35 * data['price_volume_divergence'] +
        0.25 * data['volume_cluster']
    )
    
    # Final Composite Factor
    composite_factor = (
        0.25 * momentum_decay +
        0.22 * range_breakout +
        0.20 * volatility_factor +
        0.18 * opening_imbalance +
        0.15 * price_anchoring
    )
    
    # Normalize and clean
    composite_factor = (composite_factor - composite_factor.rolling(window=20).mean()) / composite_factor.rolling(window=20).std()
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return composite_factor
