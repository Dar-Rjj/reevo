import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Efficiency Factor
    # Calculate Intraday Momentum Component
    intraday_momentum = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Assess Momentum Persistence
    momentum_lag = intraday_momentum.shift(1)
    rolling_corr = intraday_momentum.rolling(window=5).corr(momentum_lag)
    
    volume_avg_20 = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_avg_20.replace(0, np.nan)
    persistence_score = rolling_corr * volume_ratio
    
    # Combine with Range Efficiency
    daily_range_efficiency = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) == 20 else np.nan
    )
    efficiency_persistence = persistence_score * daily_range_efficiency * volume_percentile
    
    # Opening Gap Volatility Factor
    # Calculate Opening Gap Component
    opening_gap = df['open'] / df['close'].shift(1) - 1
    
    # Intraday Gap Filling Analysis
    gap_filling_pct = np.where(
        opening_gap > 0,
        (df['high'] - df['open']) / (opening_gap * df['close'].shift(1)).replace(0, np.nan),
        (df['open'] - df['low']) / (abs(opening_gap) * df['close'].shift(1)).replace(0, np.nan)
    )
    
    volume_avg_10 = df['volume'].rolling(window=10).mean()
    gap_volume_ratio = df['volume'] / volume_avg_10.replace(0, np.nan)
    gap_persistence = (1 - gap_filling_pct) * gap_volume_ratio
    
    # Volatility Regime Integration
    daily_range = df['high'] - df['low']
    volatility_regime = daily_range.rolling(window=10).std()
    volatility_quantile = volatility_regime.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) == 20 else np.nan
    )
    
    range_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    gap_volatility_factor = gap_persistence * np.where(
        volatility_quantile > 0.7,
        range_position,
        1 - abs(range_position - 0.5)
    )
    
    # Amount-Weighted Trend Reversal Factor
    # Calculate Price-Volume Trend Component
    price_change_dir = np.sign(df['close'] - df['close'].shift(1))
    amount_change_dir = np.sign(df['amount'] - df['amount'].shift(1))
    price_volume_trend = price_change_dir * amount_change_dir
    
    # Trend Strength Assessment
    trend_consistency = price_volume_trend.rolling(window=3).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 3 else np.nan
    )
    
    amount_avg_trend = df['amount'].rolling(window=3).mean()
    amount_magnitude = df['amount'] / amount_avg_trend.replace(0, np.nan)
    trend_strength = trend_consistency * amount_magnitude
    
    # Reversal Signal Integration
    intraday_volatility = df['high'] - df['low']
    reversal_factor = trend_strength * intraday_volatility * (-price_volume_trend)
    
    # Range Breakout Confirmation Factor
    # Compute Price Range Efficiency (reuse from above)
    range_efficiency = daily_range_efficiency
    
    # Volatility Context Analysis
    range_volatility = daily_range.rolling(window=10).std()
    range_volatility_quantile = range_volatility.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) == 20 else np.nan
    )
    
    breakout_condition = np.where(
        range_volatility_quantile < 0.3,
        range_efficiency,
        (df['high'] - df['high'].rolling(window=5).max()) / df['close'].shift(1)
    )
    
    # Volume-Amount Confirmation
    volume_rank = volume_percentile  # reuse from above
    
    amount_dir_3d = np.sign(df['amount'] - df['amount'].shift(1)).rolling(window=3).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 3 else np.nan
    )
    amount_avg_3d = df['amount'].rolling(window=3).mean()
    amount_magnitude_3d = df['amount'] / amount_avg_3d.replace(0, np.nan)
    amount_consistency = amount_dir_3d * amount_magnitude_3d
    
    breakout_confirmation = breakout_condition * volume_rank * amount_consistency
    
    # Combine all factors
    combined_factor = (
        efficiency_persistence.fillna(0) +
        gap_volatility_factor.fillna(0) +
        reversal_factor.fillna(0) +
        breakout_confirmation.fillna(0)
    )
    
    return combined_factor
