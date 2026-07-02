import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Momentum Divergence Component
    # Fast Momentum (EMA8)
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['fast_momentum'] = df['close'] - df['ema_fast']
    
    # Slow Momentum (EMA21)
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['slow_momentum'] = df['close'] - df['ema_slow']
    
    # Momentum Divergence
    df['momentum_divergence'] = df['fast_momentum'] - df['slow_momentum']
    
    # Volatility Weighting
    # Normalized Volatility
    df['volatility'] = df['returns'].rolling(window=14).std()
    df['normalized_volatility'] = df['volatility'].rank(pct=True)
    
    # Volume Confirmation
    df['volume_rank'] = df['volume'].rolling(window=14).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine components with volatility weighting
    df['factor'] = df['momentum_divergence'] * df['normalized_volatility'] * df['volume_rank']
    
    # Apply EMA decay (0.5, 3)
    df['factor'] = df['factor'].ewm(alpha=0.5, adjust=False).mean()
    
    return df['factor']
