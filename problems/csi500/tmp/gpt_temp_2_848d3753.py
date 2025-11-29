import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Pressure Ratio
    daily_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / daily_range.replace(0, np.nan)
    pressure_ratio = close_position.rolling(window=5).mean()
    
    # Volume-Adjusted Momentum
    price_momentum = df['close'].pct_change(periods=3)
    volume_trend = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    volume_adjusted_momentum = price_momentum * volume_trend
    
    # Opening Gap Persistence
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_filling = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    persistence_score = opening_gap * (1 - abs(gap_filling))
    
    # Volatility-Regulated Return
    daily_returns = df['close'].pct_change()
    realized_volatility = daily_returns.rolling(window=10).std()
    volatility_adjusted_return = daily_returns / (realized_volatility + 1e-8)
    
    # Amount Efficiency Ratio
    price_change = df['close'].diff()
    trading_intensity = df['amount'] / df['volume'].replace(0, np.nan)
    efficiency_ratio = price_change.rolling(window=5).sum() / trading_intensity.rolling(window=5).sum().replace(0, np.nan)
    
    # Multi-Timeframe Pressure
    short_term_compression = daily_range.rolling(window=5).std() / daily_range.rolling(window=20).std()
    medium_term_expansion = daily_range / daily_range.rolling(window=20).mean()
    timeframe_pressure = short_term_compression * medium_term_expansion
    
    # Close-Relative Strength
    volume_quantile = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    strength_alignment = close_position * volume_quantile
    
    # Intraday Momentum
    morning_momentum = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    afternoon_momentum = (df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low']).replace(0, np.nan)
    session_consistency = morning_momentum * afternoon_momentum
    
    # Price-Volume Divergence
    price_trend = df['close'].rolling(window=5).mean() / df['close'].rolling(window=20).mean()
    volume_trend_ratio = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    divergence_strength = (price_trend - volume_trend_ratio).abs()
    
    # Range Breakout Probability
    range_position = (df['close'] - df['low']) / daily_range.replace(0, np.nan)
    boundary_volume = df['volume'] / df['volume'].rolling(window=20).mean()
    breakout_likelihood = (range_position - 0.5).abs() * boundary_volume
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'pressure_ratio': pressure_ratio,
        'volume_momentum': volume_adjusted_momentum,
        'persistence': persistence_score,
        'vol_return': volatility_adjusted_return,
        'efficiency': efficiency_ratio,
        'timeframe_pressure': timeframe_pressure,
        'strength_alignment': strength_alignment,
        'session_consistency': session_consistency,
        'divergence': divergence_strength,
        'breakout_prob': breakout_likelihood
    })
    
    # Z-score normalization for each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / x.rolling(window=20).std())
    
    # Equal-weighted combination
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
