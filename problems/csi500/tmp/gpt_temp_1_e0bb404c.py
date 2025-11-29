import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Efficiency with Gap Analysis factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Dynamics Analysis
    # Gap Magnitude (Open vs Previous Close)
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Gap Filling Behavior
    # Intraday Closure Tracking
    data['intraday_high'] = data['high'].rolling(window=5, min_periods=3).max()
    data['intraday_low'] = data['low'].rolling(window=5, min_periods=3).min()
    data['gap_fill_ratio'] = np.where(
        data['gap_magnitude'] > 0,
        (data['intraday_low'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['intraday_high'] - data['prev_close']) / (data['open'] - data['prev_close'])
    )
    
    # Gap Persistence at Close
    data['gap_persistence'] = np.where(
        data['gap_magnitude'] > 0,
        (data['close'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['close']) / (data['prev_close'] - data['open'])
    )
    
    # Momentum Integration
    # Intraday Momentum Acceleration
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['momentum_accel'] = data['intraday_return'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    # Hourly Price Change Patterns (simulated with intraday high/low)
    data['price_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum Strength Assessment
    data['momentum_strength'] = data['intraday_return'].rolling(window=10, min_periods=5).apply(
        lambda x: x.mean() / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    # Gap-Momentum Interaction
    # Gap-Driven Momentum Patterns
    data['gap_momentum_interaction'] = data['gap_magnitude'] * data['momentum_strength']
    
    # Momentum Efficiency Scoring
    data['momentum_efficiency'] = data['price_range_efficiency'] * data['momentum_strength']
    
    # Liquidity Enhancement
    # Amount-Based Liquidity Signals
    data['amount_ma'] = data['amount'].rolling(window=10, min_periods=5).mean()
    data['amount_concentration'] = data['amount'] / (data['amount_ma'] + 1e-8)
    
    # Amount-Price Relationship
    data['amount_price_sensitivity'] = data['intraday_return'].rolling(window=10, min_periods=5).corr(
        data['amount_concentration'].rolling(window=10, min_periods=5).mean()
    )
    
    # Composite Alpha Generation
    # Gap-Momentum Efficiency Factor
    gap_momentum_factor = (
        data['gap_magnitude'] * data['momentum_efficiency'] * 
        (1 - abs(data['gap_fill_ratio'])) * data['gap_persistence']
    )
    
    # Liquidity-Adjusted Signal Strength
    liquidity_adjustment = np.tanh(data['amount_concentration']) * (1 + data['amount_price_sensitivity'])
    
    # Final composite factor
    alpha_factor = gap_momentum_factor * liquidity_adjustment
    
    # Clean up intermediate columns
    result = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
