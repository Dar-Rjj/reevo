import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining volatility regimes, momentum asymmetry,
    auction dynamics, liquidity efficiency, session phases, and price rejection strength.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns check
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Volatility Regime Detection
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['vol_5d'] = df['daily_range'].rolling(window=5).std()
    df['vol_20d'] = df['daily_range'].rolling(window=20).std()
    volatility_regime = (df['vol_5d'] / df['vol_20d']).fillna(0)
    
    # Volume-volatility coupling
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    volume_regime = (df['volume_ma5'] / df['volume_ma20']).fillna(0)
    vol_volume_coupling = volatility_regime * volume_regime
    
    # Momentum Asymmetry
    df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['close_to_low'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, 1e-6)
    directional_strength = (df['close_to_high'] - df['close_to_low']).fillna(0)
    
    # Price acceleration
    df['price_change'] = df['close'].pct_change()
    df['up_accel'] = df['price_change'].rolling(window=3).apply(
        lambda x: np.mean([x.iloc[i] for i in range(len(x)) if x.iloc[i] > 0]), raw=False
    ).fillna(0)
    df['down_accel'] = df['price_change'].rolling(window=3).apply(
        lambda x: np.mean([x.iloc[i] for i in range(len(x)) if x.iloc[i] < 0]), raw=False
    ).fillna(0)
    momentum_asymmetry = (df['up_accel'] - df['down_accel']).fillna(0)
    
    # Opening Auction Dynamics
    df['open_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['auction_pressure'] = df['open_gap'].rolling(window=5).mean().fillna(0)
    
    # Post-auction validation (first 30 minutes implied)
    df['early_strength'] = ((df['high'].rolling(window=3).max() - df['open']) / 
                           df['open']).fillna(0)
    
    # Liquidity Efficiency
    df['price_impact'] = (df['high'] - df['low']) / (df['volume'].replace(0, 1e-6))
    df['liquidity_efficiency'] = (1 / df['price_impact'].rolling(window=10).mean()).fillna(0)
    
    # Session Phase Analysis
    df['morning_momentum'] = ((df['close'].rolling(window=5).mean() - 
                              df['open'].rolling(window=5).mean()) / 
                             df['open'].rolling(window=5).mean()).fillna(0)
    
    # Price Rejection Strength
    df['support_test'] = ((df['low'] - df['low'].rolling(window=10).min()) / 
                         df['low'].rolling(window=10).min()).fillna(0)
    df['resistance_test'] = ((df['high'].rolling(window=10).max() - df['high']) / 
                            df['high'].rolling(window=10).max()).fillna(0)
    rejection_strength = (df['support_test'] + df['resistance_test']).fillna(0)
    
    # Combine all components with weights
    factor = (
        0.15 * volatility_regime +
        0.12 * vol_volume_coupling +
        0.18 * directional_strength +
        0.14 * momentum_asymmetry +
        0.11 * df['auction_pressure'] +
        0.10 * df['early_strength'] +
        0.08 * df['liquidity_efficiency'] +
        0.07 * df['morning_momentum'] +
        0.05 * rejection_strength
    )
    
    # Clean up temporary columns
    temp_cols = ['daily_range', 'vol_5d', 'vol_20d', 'volume_ma5', 'volume_ma20',
                'close_to_high', 'close_to_low', 'price_change', 'up_accel', 'down_accel',
                'open_gap', 'early_strength', 'price_impact', 'morning_momentum',
                'support_test', 'resistance_test']
    for col in temp_cols:
        if col in df.columns:
            del df[col]
    
    result = factor.fillna(0)
    return result
