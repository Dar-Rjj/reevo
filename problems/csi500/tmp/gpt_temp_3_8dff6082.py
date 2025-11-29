import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Momentum Divergence Factor
    # Compute Short-Term Momentum (5-day price change)
    short_momentum = data['close'].pct_change(periods=5)
    
    # Compute Medium-Term Momentum (20-day price change)
    medium_momentum = data['close'].pct_change(periods=20)
    
    # Calculate Volume Trend (5-day vs 20-day volume ratio)
    vol_5d = data['volume'].rolling(window=5).mean()
    vol_20d = data['volume'].rolling(window=20).mean()
    volume_trend = vol_5d / vol_20d
    
    # Calculate Divergence Ratio
    momentum_divergence = (short_momentum / medium_momentum) * volume_trend
    
    # Volatility-Adjusted Gap Factor
    # Calculate Price Gaps (absolute gap percentage)
    price_gap = abs(data['open'] / data['close'].shift(1) - 1)
    
    # Compute Recent Volatility (10-day average true range)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    volatility = true_range.rolling(window=10).mean()
    
    # Calculate Volume Persistence (1 - coefficient of variation)
    vol_cv = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=5).mean()
    volume_persistence = 1 - vol_cv
    
    # Create Gap-to-Volatility Ratio
    gap_vol_ratio = (price_gap / volatility) * volume_persistence
    
    # Intraday Momentum Persistence Factor
    # Calculate Intraday Strength
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    intraday_strength_3d = intraday_strength.rolling(window=3).mean()
    
    # Assess Volume Confirmation (volume change during strong intraday moves)
    strong_moves = abs(intraday_strength) > intraday_strength.rolling(window=10).std()
    volume_confirmation = data['volume'].pct_change().where(strong_moves, 0)
    
    # Calculate Price Range Consistency (1 - 5-day range variation)
    daily_range = data['high'] - data['low']
    range_cv = daily_range.rolling(window=5).std() / daily_range.rolling(window=5).mean()
    range_consistency = 1 - range_cv
    
    # Combine Signals
    intraday_persistence = intraday_strength_3d * volume_confirmation * range_consistency
    
    # Amount-Based Smart Money Indicator
    # Compute Large Trade Concentration (average trade size)
    trade_size = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Track Large Trade Trend (5-day vs 20-day average)
    trade_size_5d = trade_size.rolling(window=5).mean()
    trade_size_20d = trade_size.rolling(window=20).mean()
    trade_size_trend = trade_size_5d / trade_size_20d
    
    # Calculate Price Impact (correlation with trade size changes)
    price_returns = data['close'].pct_change()
    trade_size_changes = trade_size.pct_change()
    
    # Rolling correlation over 10 days
    corr_window = 10
    price_impact = pd.Series(index=data.index, dtype=float)
    for i in range(corr_window, len(data)):
        window_returns = price_returns.iloc[i-corr_window:i]
        window_trade_changes = trade_size_changes.iloc[i-corr_window:i]
        if len(window_returns.dropna()) > 5 and len(window_trade_changes.dropna()) > 5:
            corr = window_returns.corr(window_trade_changes)
            price_impact.iloc[i] = corr if not pd.isna(corr) else 0
        else:
            price_impact.iloc[i] = 0
    
    # Adjust for Market Phase (overall price trend)
    market_trend = data['close'].pct_change(periods=20)
    trend_weight = 1 + abs(market_trend)  # Higher weight in trending markets
    
    # Create Smart Money Signal
    smart_money = trade_size_trend * price_impact * trend_weight
    
    # Range Breakout Efficiency Factor
    # Identify Breakout Events (new 10-day highs/lows)
    rolling_high = data['high'].rolling(window=10).max()
    rolling_low = data['low'].rolling(window=10).min()
    
    high_breakout = data['high'] > rolling_high.shift(1)
    low_breakout = data['low'] < rolling_low.shift(1)
    breakout_events = high_breakout | low_breakout
    
    # Measure Breakout Quality (follow-through percentage)
    breakout_direction = np.where(high_breakout, 1, np.where(low_breakout, -1, 0))
    close_position = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 0.5)
    follow_through = abs(close_position - 0.5) * breakout_direction
    
    # Calculate Volume Support (volume-to-range ratio)
    volume_range_ratio = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Create Efficiency Score
    breakout_efficiency = breakout_events.astype(float) * follow_through * volume_range_ratio
    
    # Combine all factors with equal weighting
    factor = (
        momentum_divergence.fillna(0) +
        gap_vol_ratio.fillna(0) +
        intraday_persistence.fillna(0) +
        smart_money.fillna(0) +
        breakout_efficiency.fillna(0)
    )
    
    return factor
