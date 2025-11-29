import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Volatility Skewness
    daily_range = df['high'] - df['low']
    range_skew = daily_range.rolling(window=20, min_periods=10).skew()
    
    # Volume-Adjusted Price Momentum Divergence
    price_momentum = df['close'].pct_change(periods=5)
    volume_trend = df['volume'].rolling(window=5).mean().pct_change(periods=5)
    vol_adj_momentum_div = price_momentum - volume_trend
    
    # Opening Gap Persistence
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_direction_consistency = opening_gap.rolling(window=5).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan
    )
    
    # Intraday Reversal Strength
    intraday_movement = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    volume_concentration = df['volume'] / df['volume'].rolling(window=5).mean()
    intraday_reversal = intraday_movement * volume_concentration
    
    # Price-Volume Acceleration Divergence
    price_acceleration = df['close'].pct_change(periods=3).diff(periods=2)
    volume_acceleration = df['volume'].pct_change(periods=3).diff(periods=2)
    accel_divergence = price_acceleration - volume_acceleration
    
    # Resistance Breakout Confirmation
    resistance_level = df['high'].rolling(window=20).max().shift(1)
    breakout = (df['close'] > resistance_level).astype(int)
    breakout_volume = df['volume'] / df['volume'].rolling(window=20).mean()
    breakout_confirmation = breakout * breakout_volume
    
    # Low-Volume Trend Continuation
    volume_quantile = df['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan
    )
    low_volume_periods = (volume_quantile < 0.3).astype(int)
    trend_persistence = df['close'].pct_change(periods=3).rolling(window=5).std()
    low_volume_trend = low_volume_periods * (1 / (trend_persistence + 1e-6))
    
    # Amplitude-Volume Correlation Reversal
    price_amplitude = (df['high'] - df['low']) / df['close']
    amplitude_volume_corr = price_amplitude.rolling(window=10).corr(df['volume'])
    corr_reversal = -amplitude_volume_corr.diff(periods=3)
    
    # Close Position Strength
    close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    volume_pattern = df['volume'] / df['volume'].rolling(window=5).mean()
    close_strength = close_position * volume_pattern
    
    # Multi-timeframe Momentum Alignment
    short_term_momentum = df['close'].pct_change(periods=3)
    medium_term_momentum = df['close'].pct_change(periods=10)
    momentum_alignment = short_term_momentum - medium_term_momentum
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'range_skew': range_skew,
        'vol_adj_momentum_div': vol_adj_momentum_div,
        'gap_direction_consistency': gap_direction_consistency,
        'intraday_reversal': intraday_reversal,
        'accel_divergence': accel_divergence,
        'breakout_confirmation': breakout_confirmation,
        'low_volume_trend': low_volume_trend,
        'corr_reversal': corr_reversal,
        'close_strength': close_strength,
        'momentum_alignment': momentum_alignment
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=60, min_periods=20).mean()) / 
                                      x.rolling(window=60, min_periods=20).std())
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
