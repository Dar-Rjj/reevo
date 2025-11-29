import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Momentum Acceleration
    # Calculate Short-Term Momentum
    mom_5d = data['close'].pct_change(5)
    mom_10d = data['close'].pct_change(10)
    
    # Calculate Momentum Acceleration
    mom_accel = (mom_5d - mom_10d) / (data['close'].abs() + 1e-8)
    
    # Volume Confirmation
    vol_5d_avg = data['volume'].rolling(5).mean()
    vol_trend = data['volume'] / (vol_5d_avg + 1e-8)
    factor1 = mom_accel * vol_trend
    
    # High-Low Range Breakout
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    
    # Identify Range Breakouts
    high_5d_max = data['high'].rolling(5).max()
    low_5d_min = data['low'].rolling(5).min()
    high_breakout = (data['high'] - high_5d_max.shift(1)) / (high_5d_max.shift(1) + 1e-8)
    low_breakout = (low_5d_min.shift(1) - data['low']) / (low_5d_min.shift(1) + 1e-8)
    breakout_magnitude = np.where(high_breakout > low_breakout, high_breakout, -low_breakout)
    
    # Volume-Weighted Breakout Strength
    factor2 = breakout_magnitude * data['volume'] / (atr + 1e-8)
    
    # Opening Gap Reversal
    # Calculate Opening Gap
    gap = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Identify Overreaction
    hist_vol = data['close'].pct_change().rolling(20).std()
    intraday_move = (data['close'] - data['open']) / (data['open'] + 1e-8)
    
    # Reversal Signal
    reversal_signal = -gap * intraday_move * data['volume']
    factor3 = reversal_signal / (hist_vol + 1e-8)
    
    # Intraday Momentum Divergence
    # Calculate Morning Session Momentum
    morning_up = (data['high'] - data['open']) / (data['open'] + 1e-8)
    morning_down = (data['open'] - data['low']) / (data['open'] + 1e-8)
    morning_momentum = morning_up - morning_down
    
    # Calculate Afternoon Session Momentum
    mid_price = (data['high'] + data['low']) / 2
    afternoon_momentum = (data['close'] - mid_price) / (mid_price + 1e-8)
    
    momentum_divergence = morning_momentum - afternoon_momentum
    
    # Volume Pattern Analysis
    morning_volume_ratio = data['volume'].rolling(10).apply(lambda x: x[:5].sum() / (x[5:].sum() + 1e-8))
    factor4 = momentum_divergence * morning_volume_ratio
    
    # Price-Volume Efficiency
    # Calculate Price Movement Efficiency
    hl_range = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    cc_move = abs(data['close'].pct_change())
    efficiency_ratio = cc_move / (hl_range + 1e-8)
    
    # Volume Distribution Analysis
    volume_spike = data['volume'] / data['volume'].rolling(20).mean()
    volume_persistence = data['volume'].rolling(5).std() / (data['volume'].rolling(5).mean() + 1e-8)
    
    # Combined Signal
    factor5 = efficiency_ratio * volume_persistence * volume_spike
    
    # Volatility Regime Switching
    # Identify Volatility Regimes
    short_vol = data['close'].pct_change().rolling(5).std()
    medium_vol = data['close'].pct_change().rolling(20).std()
    vol_ratio = short_vol / (medium_vol + 1e-8)
    
    # Price Behavior in Different Regimes
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    
    # Adaptive Factor
    regime_factor = np.where(high_vol_regime, -mom_5d, 
                           np.where(low_vol_regime, mom_5d, 0))
    factor6 = regime_factor * data['volume'] / (data['volume'].rolling(20).mean() + 1e-8)
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5,
        'factor6': factor6
    })
    
    # Normalize each factor by its rolling z-score
    final_factor = pd.Series(index=data.index, dtype=float)
    for col in factors.columns:
        normalized = (factors[col] - factors[col].rolling(60).mean()) / (factors[col].rolling(60).std() + 1e-8)
        final_factor = final_factor.add(normalized.fillna(0), fill_value=0)
    
    return final_factor / 6  # Average of 6 factors
