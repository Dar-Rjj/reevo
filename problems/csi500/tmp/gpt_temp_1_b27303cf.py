import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Elasticity-Momentum Convergence Dynamics Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price-based features
    df['prev_close'] = df.groupby(level=1)['close'].shift(1)
    df['prev_high'] = df.groupby(level=1)['high'].shift(1)
    df['prev_low'] = df.groupby(level=1)['low'].shift(1)
    
    # Overnight gap and momentum
    df['overnight_return'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Price sensitivity to volume shocks (elasticity proxy)
    df['volume_change'] = df.groupby(level=1)['volume'].pct_change()
    df['price_change'] = df.groupby(level=1)['close'].pct_change()
    df['price_sensitivity'] = df['price_change'].rolling(window=5, min_periods=3).corr(df['volume_change'])
    
    # Elasticity-adjusted gap
    df['elasticity_adjusted_gap'] = df['overnight_return'] * df['price_sensitivity']
    
    # Intraday range analysis
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    df['prev_range'] = df.groupby(level=1)['daily_range'].shift(1)
    
    # Range completion speed proxy (using open-to-high time as range achievement)
    df['range_achievement'] = (df['high'] - df['open']) / (df['high'] - df['low'])
    df['range_completion_speed'] = df['range_achievement'].rolling(window=5, min_periods=3).mean()
    
    # Bounce recovery strength
    df['bounce_recovery'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Session momentum proxies
    df['open_to_high_ratio'] = (df['high'] - df['open']) / df['open']
    df['low_to_close_ratio'] = (df['close'] - df['low']) / df['low']
    df['session_transition_momentum'] = df['low_to_close_ratio'] - df['open_to_high_ratio']
    
    # Volume concentration patterns
    df['volume_ma_5'] = df.groupby(level=1)['volume'].rolling(window=5, min_periods=3).mean().reset_index(level=1, drop=True)
    df['volume_skew'] = df['volume'] / df['volume_ma_5']
    
    # Volatility analysis
    df['morning_volatility'] = (df['high'] - df['low']) / df['open']  # Simplified proxy
    df['prev_volatility'] = df.groupby(level=1)['morning_volatility'].shift(1)
    df['volatility_expansion'] = df['morning_volatility'] / df['prev_volatility']
    
    # Flow acceleration patterns (using amount as proxy)
    df['amount_change'] = df.groupby(level=1)['amount'].pct_change()
    df['flow_acceleration'] = df['amount_change'].rolling(window=5, min_periods=3).mean()
    
    # Regime classification
    df['volatility_regime'] = (df['morning_volatility'] > df['morning_volatility'].rolling(window=20, min_periods=10).median()).astype(int)
    df['elasticity_regime'] = (df['price_sensitivity'] > df['price_sensitivity'].rolling(window=20, min_periods=10).median()).astype(int)
    
    # Component factors
    df['elasticity_momentum_factor'] = df['elasticity_adjusted_gap'] * df['session_transition_momentum']
    df['range_efficiency_factor'] = df['range_completion_speed'] * df['bounce_recovery']
    df['volume_concentration_factor'] = df['volume_skew'] * df['flow_acceleration']
    
    # Volatility regime factor with regime consistency
    df['regime_consistency'] = (df['volatility_regime'] == df['elasticity_regime']).astype(float)
    df['volatility_regime_factor'] = df['volatility_expansion'] * df['regime_consistency']
    
    # Composite alpha construction
    alpha = (
        df['elasticity_momentum_factor'] * 
        df['range_efficiency_factor'] * 
        df['volume_concentration_factor'] * 
        df['volatility_regime_factor']
    )
    
    # Clean up and forward fill missing values
    alpha = alpha.groupby(level=1).ffill()
    
    return alpha
