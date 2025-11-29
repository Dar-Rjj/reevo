import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Range Breakout Momentum
    daily_range = data['high'] - data['low']
    range_percentile = daily_range.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.5)) / (x.quantile(0.8) - x.quantile(0.2)) if len(x.dropna()) >= 10 else np.nan, 
        raw=False
    )
    volume_ratio = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    breakout_momentum = range_percentile * volume_ratio
    
    # Volume-Adjusted Intraday Reversal
    high_open_ratio = (data['high'] - data['open']) / data['open']
    low_open_ratio = (data['low'] - data['open']) / data['open']
    intraday_extreme = np.where(
        abs(high_open_ratio) > abs(low_open_ratio),
        high_open_ratio,
        low_open_ratio
    )
    intraday_extreme_percentile = pd.Series(intraday_extreme, index=data.index).rolling(
        window=10, min_periods=5
    ).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x.dropna()) >= 5 and x.std() > 0 else 0,
        raw=False
    )
    volume_persistence = data['volume'].rolling(window=5, min_periods=3).std() / data['volume'].rolling(window=5, min_periods=3).mean()
    intraday_reversal = -intraday_extreme_percentile * (1 + volume_persistence)
    
    # Amount-Driven Price Impact
    price_change = data['close'].pct_change()
    amount_change = data['amount'].pct_change()
    price_per_amount = price_change / (amount_change + 1e-8)
    slippage_pattern = price_per_amount.rolling(window=15, min_periods=8).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x.dropna()) >= 8 and x.std() > 0 else 0,
        raw=False
    )
    amount_impact = -slippage_pattern.rolling(window=5, min_periods=3).mean()
    
    # Multi-Timeframe Momentum Divergence
    short_momentum = data['close'].pct_change(periods=3)
    medium_momentum = data['close'].pct_change(periods=10)
    momentum_divergence = short_momentum - medium_momentum
    momentum_divergence_z = momentum_divergence.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x.dropna()) >= 10 and x.std() > 0 else 0,
        raw=False
    )
    volume_confirmation = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    momentum_factor = momentum_divergence_z * volume_confirmation
    
    # Open-Gap Persistence
    prev_close = data['close'].shift(1)
    opening_gap = (data['open'] - prev_close) / prev_close
    intraday_range = (data['high'] - data['low']) / data['open']
    gap_persistence = abs(opening_gap) / (intraday_range + 1e-8)
    gap_adjustment = -opening_gap * gap_persistence
    volume_support = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    gap_factor = gap_adjustment * volume_support
    
    # Combine all factors with equal weights
    combined_factor = (
        breakout_momentum.fillna(0) +
        intraday_reversal.fillna(0) +
        amount_impact.fillna(0) +
        momentum_factor.fillna(0) +
        gap_factor.fillna(0)
    )
    
    # Normalize the final factor
    final_factor = (combined_factor - combined_factor.rolling(window=20, min_periods=10).mean()) / (
        combined_factor.rolling(window=20, min_periods=10).std() + 1e-8
    )
    
    return final_factor
