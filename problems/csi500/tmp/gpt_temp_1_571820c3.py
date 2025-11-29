import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns and ranges
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['open_range'] = (df['open'] - df['low'].shift(1)) / df['close'].shift(1)
    df['close_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Bull/Bear volume fractal dimensions
    df['up_day'] = (df['returns'] > 0).astype(int)
    df['down_day'] = (df['returns'] < 0).astype(int)
    
    # Calculate volume-weighted metrics
    df['vwap'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Multi-fractal price-volume asymmetry
    df['bull_volume_fractal'] = df['volume'] * df['up_day']
    df['bear_volume_fractal'] = df['volume'] * df['down_day']
    
    # Rolling fractal dimensions (5-day and 20-day)
    for window in [5, 20]:
        df[f'bull_frac_{window}'] = df['bull_volume_fractal'].rolling(window=window).std() / df['bull_volume_fractal'].rolling(window=window).mean()
        df[f'bear_frac_{window}'] = df['bear_volume_fractal'].rolling(window=window).std() / df['bear_volume_fractal'].rolling(window=window).mean()
    
    # Volume asymmetry persistence
    df['volume_asymmetry'] = (df['bull_volume_fractal'].rolling(window=5).mean() - 
                             df['bear_volume_fractal'].rolling(window=5).mean())
    df['asymmetry_persistence'] = df['volume_asymmetry'].rolling(window=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0)
    
    # Price-volume fractal divergence
    df['price_momentum_5'] = df['close'].pct_change(5)
    df['volume_momentum_5'] = df['volume'].pct_change(5)
    df['fractal_divergence'] = (df['price_momentum_5'] - df['volume_momentum_5']) * df['volume_ratio']
    
    # Intraday range expansion asymmetry
    df['morning_range'] = (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()) / df['close'].shift(3)
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['range_efficiency'] = df['range_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Multi-scale range efficiency
    df['short_range_eff'] = df['range_efficiency'].rolling(window=5).mean()
    df['long_range_eff'] = df['range_efficiency'].rolling(window=20).mean()
    df['range_efficiency_score'] = df['short_range_eff'] - df['long_range_eff']
    
    # Volume-weighted momentum validation
    df['heavy_volume_momentum'] = df['returns'] * np.log1p(df['volume_ratio'])
    df['momentum_persistence'] = df['heavy_volume_momentum'].rolling(window=10).apply(
        lambda x: len(np.where(np.diff(np.sign(x)) != 0)[0]) / max(len(x)-1, 1))
    
    # Final factor combination
    factor = (
        0.3 * df['fractal_divergence'] +
        0.25 * df['range_efficiency_score'] +
        0.2 * df['asymmetry_persistence'] +
        0.15 * df['heavy_volume_momentum'] -
        0.1 * df['momentum_persistence']
    )
    
    # Normalize and clean
    factor = (factor - factor.rolling(window=60).mean()) / factor.rolling(window=60).std()
    factor = factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return factor
