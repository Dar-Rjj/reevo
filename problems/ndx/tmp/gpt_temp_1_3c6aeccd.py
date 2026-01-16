import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Momentum Component
    # Intraday Price Momentum
    df['intraday_momentum'] = (df['high'] - df['low']) / df['open']
    
    # Historical Momentum
    df['daily_return'] = df['close'].pct_change()
    df['historical_momentum'] = df['daily_return'].rolling(window=5).mean()
    
    # Liquidity Component
    # Relative Volume Strength
    df['ma_volume_20'] = df['volume'].rolling(window=20).mean()
    df['relative_volume_strength'] = df['volume'] / df['ma_volume_20']
    
    # Relative Turnover Strength
    df['amount'] = df['volume'] * df['close']
    df['ma_amount_20'] = df['amount'].rolling(window=20).mean()
    df['relative_turnover_strength'] = df['amount'] / df['ma_amount_20']
    
    # Signal Integration
    # Multiply Intraday Price Momentum by Relative Volume Strength
    df['momentum_volume'] = df['intraday_momentum'] * df['relative_volume_strength']
    
    # Multiply Historical Momentum by Relative Turnover Strength
    df['momentum_turnover'] = df['historical_momentum'] * df['relative_turnover_strength']
    
    # Combine signals
    df['combined_signal'] = df['momentum_volume'] + df['momentum_turnover']
    
    # Normalize by Cross-Sectional Z-score
    df['factor'] = df.groupby(df.index)['combined_signal'].transform(lambda x: zscore(x))
    
    return df['factor']
