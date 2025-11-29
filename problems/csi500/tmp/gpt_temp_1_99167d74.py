import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    vol_ratio = (returns.rolling(3).std().ewm(span=5).mean() / 
                 returns.rolling(10).std().ewm(span=10).mean())
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    
    # True Range and ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_5 = true_range.rolling(5).mean()
    
    # Breakout Strength Assessment
    rolling_high_5 = df['high'].rolling(5).max()
    rolling_low_5 = df['low'].rolling(5).min()
    breakout_ratio_high = (df['high'] - rolling_high_5) / (rolling_high_5 - rolling_low_5)
    breakout_ratio_low = (df['low'] - rolling_low_5) / (rolling_high_5 - rolling_low_5)
    breakout_ratio = (breakout_ratio_high - breakout_ratio_low).fillna(0)
    
    vol_persistence = (df['close'].rolling(5).std() / df['close'].rolling(10).std()).fillna(1)
    
    # Intraday Momentum & Reversal Synthesis
    intraday_momentum = (df['close'] - df['open']) / (df['open'] * true_range)
    intraday_momentum_smoothed = intraday_momentum.rolling(3).mean()
    intraday_momentum_adaptive = np.where(low_vol_regime, intraday_momentum_smoothed, intraday_momentum)
    
    open_to_close_return = (df['close'] - df['open']) / df['open']
    prev_close_to_open_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    reversal_indicator = -np.sign(prev_close_to_open_return) * open_to_close_return
    
    vol_adjusted_returns = returns / atr_5
    
    # Momentum Direction Analysis
    midpoint = (df['high'] + df['low']) / 2
    momentum_direction = np.where(df['close'] > midpoint, 1, -1)
    vol_adjusted_momentum = momentum_direction / true_range
    
    # Volume-Price Confirmation & Divergence
    volume_20d_avg = df['volume'].rolling(20).mean()
    volume_5d_avg = df['volume'].rolling(5).mean()
    
    volume_divergence_signal = (abs((df['close'] - df['open']) / df['open']) * 
                               abs(df['volume'] / volume_20d_avg - 1))
    
    volume_weighted_range_efficiency = ((df['close'] - df['low']) / (df['high'] - df['low']) * 
                                       df['volume'] / volume_5d_avg).fillna(0)
    
    # Volume-Price Divergence Analysis
    volume_zscore = ((df['volume'] - df['volume'].rolling(20).mean()) / 
                     df['volume'].rolling(20).std()).fillna(0)
    
    volume_weighted_price_range = (df['high'] - df['low']) * df['volume']
    amount_efficiency = df['amount'] / (df['high'] - df['low']).replace(0, np.nan)
    amount_efficiency = amount_efficiency.fillna(0)
    
    volume_acceleration = df['volume'] / df['volume'].shift(1)
    volume_growth_3d = (df['volume'] / df['volume'].shift(3)) - 1
    
    # Divergence Scoring
    return_sign = np.sign(returns)
    volume_dev_sign = np.sign(df['volume'] - volume_20d_avg)
    divergence_score = np.where(return_sign == volume_dev_sign, 1, -1) * abs(volume_zscore)
    
    liquidity_scaled_signal = (reversal_indicator * amount_efficiency / 
                              volume_weighted_price_range.replace(0, np.nan)).fillna(0)
    
    # Multi-Timeframe Momentum Alignment
    momentum_alignment = (np.sign(df['close'] / df['close'].shift(3) - 1) * 
                         np.sign((df['high'] - df['low']) / 
                                (df['high'].shift(3) - df['low'].shift(3)) - 1))
    
    price_momentum_3d = df['close'] / df['close'].shift(3) - 1
    range_momentum_3d = ((df['high'] - df['low']) - 
                        (df['high'].shift(3) - df['low'].shift(3))) / (df['high'].shift(3) - df['low'].shift(3))
    momentum_alignment_integrated = np.sign(price_momentum_3d) * np.sign(range_momentum_3d.fillna(0))
    
    # Breakout Persistence Context
    breakout_history = (df['high'] > rolling_high_5.shift(1)).rolling(10).sum()
    breakout_persistence = breakout_history / 10
    
    volume_momentum_confirmation = np.where(
        (volume_acceleration > 1) == (momentum_alignment > 0), 1, -1
    )
    
    # Support/Resistance & Range Context
    resistance_10d = df['high'].rolling(10).max()
    support_10d = df['low'].rolling(10).min()
    proximity_score = pd.concat([
        (resistance_10d - df['close']) / df['close'],
        (df['close'] - support_10d) / df['close']
    ], axis=1).min(axis=1)
    
    current_range = df['high'] - df['low']
    prev_range = current_range.shift(1)
    range_expansion = current_range > prev_range
    consecutive_expansion = range_expansion.rolling(3).sum()
    
    # Adaptive Signal Synthesis
    volatility_breakout_component = (vol_adjusted_returns * reversal_indicator * 
                                   breakout_ratio * vol_persistence)
    
    volume_confirmation_enhancement = (volatility_breakout_component * 
                                     volume_divergence_signal * volume_acceleration.fillna(1))
    
    momentum_alignment_integration = (volume_confirmation_enhancement * 
                                    momentum_alignment_integrated * breakout_persistence)
    
    range_context_finalization = (momentum_alignment_integration * 
                                consecutive_expansion * proximity_score)
    
    # Regime-Dependent Weighting
    high_vol_weights = [0.35, 0.25, 0.20, 0.15, 0.05]
    low_vol_weights = [0.20, 0.40, 0.15, 0.20, 0.05]
    
    components = [
        intraday_momentum_adaptive,
        volume_weighted_range_efficiency + liquidity_scaled_signal,
        reversal_indicator * vol_adjusted_momentum,
        momentum_alignment * volume_momentum_confirmation,
        proximity_score * consecutive_expansion
    ]
    
    # Apply regime-specific weighting
    final_factor = pd.Series(0, index=df.index)
    for i, component in enumerate(components):
        high_vol_component = component * high_vol_weights[i] * high_vol_regime
        low_vol_component = component * low_vol_weights[i] * low_vol_regime
        final_factor += high_vol_component.fillna(0) + low_vol_component.fillna(0)
    
    # Add the range context finalization
    final_factor += range_context_finalization.fillna(0)
    
    # Percentile-based clipping
    lower_bound = final_factor.rolling(20).apply(lambda x: np.percentile(x.dropna(), 10), raw=False)
    upper_bound = final_factor.rolling(20).apply(lambda x: np.percentile(x.dropna(), 90), raw=False)
    
    final_factor_clipped = final_factor.clip(lower=lower_bound, upper=upper_bound)
    
    return final_factor_clipped
