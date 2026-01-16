import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Price Divergence Component
    df['sma_close_3'] = df['close'].rolling(window=3, min_periods=1).mean()
    df['price_divergence_short'] = df['close'] / df['sma_close_3']
    
    df['sma_close_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['price_divergence_long'] = df['close'] / df['sma_close_10']
    
    # Volume Divergence Component
    df['sma_volume_3'] = df['volume'].rolling(window=3, min_periods=1).mean()
    df['volume_divergence_short'] = df['volume'] / df['sma_volume_3']
    
    df['sma_volume_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_divergence_long'] = df['volume'] / df['sma_volume_10']
    
    # Volatility Adjustment
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combine Price and Volume Divergence
    df['price_volume_divergence'] = (df['price_divergence_short'] * df['price_divergence_long'] *
                                     df['volume_divergence_short'] * df['volume_divergence_long'])
    
    # Normalize Divergence with Volatility
    df['divergence_volatility_adjusted'] = df['price_volume_divergence'] * df['intraday_volatility']
    
    # Apply Z-score Normalization (21d rolling)
    df['divergence_normalized'] = df['divergence_volatility_adjusted'].rolling(window=21, min_periods=1).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Efficiency Confirmation
    df['normalized_price_range'] = (df['high'] - df['low']) / df['open']
    
    # Combine Signals
    df['final_signal'] = df['divergence_normalized'] * df['normalized_price_range']
    
    # Normalize by Average Efficiency Ratio (past 5 days)
    df['efficiency_ratio'] = df['normalized_price_range'].rolling(window=5, min_periods=1).mean()
    df['final_factor'] = df['final_signal'] / df['efficiency_ratio']
    
    return df['final_factor']
