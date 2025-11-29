import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Decay Factor
    # Calculate Intraday Momentum
    intraday_momentum = (data['high'] - data['low']) / data['low']
    
    # Calculate Price Persistence
    daily_returns = data['close'].pct_change()
    direction = np.sign(daily_returns)
    persistence_count = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if direction.iloc[i] == direction.iloc[i-1] and not np.isnan(direction.iloc[i]) and not np.isnan(direction.iloc[i-1]):
            persistence_count.iloc[i] = persistence_count.iloc[i-1] + 1
    
    # Combine Momentum and Persistence
    momentum_persistence = intraday_momentum * (persistence_count + 1)
    decay_factor = np.exp(-persistence_count / 10)  # Exponential decay
    intraday_factor = momentum_persistence * decay_factor
    intraday_factor = intraday_factor * (data['amount'] / data['amount'].rolling(20, min_periods=1).mean())
    
    # Volume-Adjusted Price Acceleration
    # Calculate Price Acceleration
    returns = data['close'].pct_change()
    first_diff = returns.diff()
    acceleration = first_diff.diff()
    
    # Calculate Volume Trend
    volume_ma = data['volume'].rolling(10, min_periods=1).mean()
    volume_momentum = data['volume'] / volume_ma
    
    # Combine Acceleration and Volume
    volume_acceleration = acceleration * volume_momentum
    
    # Sign correction based on direction consistency
    sign_correction = np.where(
        (acceleration > 0) & (volume_momentum > 1) & (returns > 0), 1,
        np.where((acceleration < 0) & (volume_momentum > 1) & (returns < 0), 1, 0.5)
    )
    volume_acceleration = volume_acceleration * sign_correction
    
    # Volatility clustering adjustment
    vol_20d = returns.rolling(20, min_periods=1).std()
    vol_adjustment = 1 / (1 + vol_20d)
    volume_acceleration = volume_acceleration * vol_adjustment
    
    # Relative Strength Breakout Factor
    # Calculate Relative Strength
    rolling_rank = data['close'].rolling(20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) == 20 else 0.5
    )
    
    # Identify Breakout Conditions
    resistance = data['high'].rolling(20, min_periods=1).max()
    support = data['low'].rolling(20, min_periods=1).min()
    
    breakout_up = (data['close'] > resistance.shift(1)) & (data['close'] > data['open'])
    breakout_down = (data['close'] < support.shift(1)) & (data['close'] < data['open'])
    
    breakout_magnitude = np.where(
        breakout_up, (data['close'] - resistance.shift(1)) / resistance.shift(1),
        np.where(breakout_down, (support.shift(1) - data['close']) / support.shift(1), 0)
    )
    
    # Combine Strength and Breakout
    strength_breakout = rolling_rank * breakout_magnitude
    volume_confirmation = data['volume'] / data['volume'].rolling(20, min_periods=1).mean()
    strength_breakout = strength_breakout * volume_confirmation
    
    # False breakout probability adjustment
    recent_volatility = data['high'].rolling(5, min_periods=1).std() / data['close'].rolling(5, min_periods=1).mean()
    false_breakout_adj = 1 / (1 + recent_volatility)
    strength_breakout = strength_breakout * false_breakout_adj
    
    # Amount-Weighted Price Efficiency
    # Calculate Price Efficiency
    realized_vol = (data['high'] - data['low']) / data['close']
    potential_vol = (data['high'].rolling(5, min_periods=1).max() - data['low'].rolling(5, min_periods=1).min()) / data['close']
    price_efficiency = 1 - (realized_vol / potential_vol.replace(0, 1))
    
    # Calculate Trading Intensity
    amount_per_share = data['amount'] / data['volume'].replace(0, 1)
    trading_intensity = amount_per_share / amount_per_share.rolling(20, min_periods=1).mean()
    
    # Combine Efficiency and Intensity
    efficiency_factor = price_efficiency * trading_intensity
    
    # Market regime adjustment
    market_trend = data['close'].pct_change(10).rolling(5, min_periods=1).mean()
    regime_adjustment = 1 + 0.5 * np.tanh(market_trend * 10)
    efficiency_factor = efficiency_factor * regime_adjustment
    
    # Liquidity scaling
    avg_amount = data['amount'].rolling(20, min_periods=1).mean()
    liquidity_scale = 1 / (1 + np.log1p(avg_amount / avg_amount.median()))
    efficiency_factor = efficiency_factor * liquidity_scale
    
    # Multi-Timeframe Momentum Convergence
    # Calculate Short-term Momentum
    short_momentum_1d = data['close'].pct_change(1)
    short_momentum_3d = data['close'].pct_change(3)
    short_momentum = (short_momentum_1d + short_momentum_3d) / 2
    
    # Calculate Medium-term Momentum
    medium_momentum_5d = data['close'].pct_change(5)
    medium_momentum_10d = data['close'].pct_change(10)
    medium_momentum = (medium_momentum_5d + medium_momentum_10d) / 2
    
    # Detect Convergence Patterns
    convergence_sign = np.sign(short_momentum) == np.sign(medium_momentum)
    convergence_strength = np.abs(short_momentum * medium_momentum)
    
    convergence_factor = np.where(convergence_sign, convergence_strength, -convergence_strength)
    
    # Volume confirmation
    volume_confirmation_mtf = data['volume'] / data['volume'].rolling(10, min_periods=1).mean()
    convergence_factor = convergence_factor * volume_confirmation_mtf
    
    # Momentum duration adjustment
    momentum_duration = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if convergence_sign.iloc[i] and convergence_sign.iloc[i-1]:
            momentum_duration.iloc[i] = momentum_duration.iloc[i-1] + 1
    
    duration_adjustment = 1 / (1 + momentum_duration / 10)
    convergence_factor = convergence_factor * duration_adjustment
    
    # Combine all factors with equal weighting
    final_factor = (
        intraday_factor.fillna(0) + 
        volume_acceleration.fillna(0) + 
        strength_breakout.fillna(0) + 
        efficiency_factor.fillna(0) + 
        convergence_factor.fillna(0)
    ) / 5
    
    return final_factor
