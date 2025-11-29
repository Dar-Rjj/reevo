import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily ranges and returns
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['open_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Volatility regime detection
    df['short_term_vol'] = df['daily_range'].rolling(window=5, min_periods=3).mean()
    df['long_term_vol'] = df['daily_range'].rolling(window=20, min_periods=10).mean()
    df['vol_regime'] = (df['short_term_vol'] > df['long_term_vol'] * 1.2).astype(int)
    
    # High volatility regime features
    df['gap_strength'] = (df['open_gap'].abs() * np.sign(df['open_gap'])) / df['short_term_vol']
    df['early_vs_late'] = ((df['high'] - df['open']) - (df['close'] - df['low'])) / df['open']
    
    # Maximum drawdown recovery
    df['intraday_high'] = df['high'].rolling(window=5, min_periods=3).max()
    df['recovery_strength'] = (df['close'] - df['low']) / (df['intraday_high'] - df['low']).replace(0, np.nan)
    
    # Low volatility regime features
    df['recent_high'] = df['high'].rolling(window=10, min_periods=5).max()
    df['breakout_strength'] = (df['close'] - df['recent_high']) / df['recent_high']
    df['volume_expansion'] = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    # Trend persistence
    df['price_trend'] = df['close'].rolling(window=3).apply(lambda x: 1 if (x.diff().dropna() > 0).all() else (-1 if (x.diff().dropna() < 0).all() else 0), raw=False)
    df['trend_persistence'] = df['price_trend'].rolling(window=5).sum()
    
    # Price efficiency scoring
    df['high_close_slippage'] = (df['high'] - df['close']) / (df['high'] - df['open']).replace(0, np.nan)
    df['low_close_slippage'] = (df['close'] - df['low']) / (df['open'] - df['low']).replace(0, np.nan)
    df['path_efficiency'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume-price coordination
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['volume_price_sync'] = (df['close'] - df['vwap']) / df['close'].rolling(window=5).std()
    
    # Volume concentration
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    
    # Calculate regime-specific scores
    high_vol_score = (
        df['gap_strength'] * 0.3 +
        df['early_vs_late'] * 0.4 +
        df['recovery_strength'] * 0.3
    )
    
    low_vol_score = (
        df['breakout_strength'] * 0.4 +
        df['volume_expansion'] * 0.2 +
        df['trend_persistence'] * 0.4
    )
    
    # Efficiency and coordination components
    efficiency_score = (
        (1 - df['high_close_slippage'].abs()) * 0.3 +
        (1 - df['low_close_slippage'].abs()) * 0.3 +
        df['path_efficiency'] * 0.4
    )
    
    coordination_score = (
        df['volume_price_sync'] * 0.6 +
        df['volume_zscore'] * 0.4
    )
    
    # Combine all components with regime weighting
    factor_values = (
        (df['vol_regime'] * high_vol_score + (1 - df['vol_regime']) * low_vol_score) * 0.5 +
        efficiency_score * 0.3 +
        coordination_score * 0.2
    )
    
    # Fill initial NaN values and ensure no forward-looking
    factor_values = factor_values.fillna(method='ffill').fillna(0)
    
    return factor_values
