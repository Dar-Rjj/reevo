import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    short_term_vol = daily_range.rolling(window=3, min_periods=2).std()
    medium_term_vol = daily_range.rolling(window=10, min_periods=5).std()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Volatility regime threshold (using median)
    high_vol_regime = volatility_ratio > volatility_ratio.rolling(window=20, min_periods=10).median()
    
    # Regime-Adaptive Breakout Component
    breakout_signals = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Regime: Morning expansion × Afternoon compression
    # Using first 2 hours as morning, last 2 hours as afternoon (approximated)
    morning_expansion = (data['high'].rolling(window=2).max() - data['open']) / data['open']
    afternoon_compression = (data['close'] - data['low'].rolling(window=2).min()) / data['close']
    high_vol_breakout = morning_expansion * afternoon_compression
    
    # Low Volatility Regime: Range-based breakout
    rolling_high = data['high'].rolling(window=5, min_periods=3).max()
    midpoint = (data['high'] + data['low']) / 2
    low_vol_breakout = (midpoint - rolling_high) / rolling_high
    
    # Apply regime-specific breakout signals
    breakout_signals[high_vol_regime] = high_vol_breakout[high_vol_regime]
    breakout_signals[~high_vol_regime] = low_vol_breakout[~high_vol_regime]
    
    # Apply 3-day moving average to breakout signals
    breakout_component = breakout_signals.rolling(window=3, min_periods=2).mean()
    
    # Regime-Adaptive Reversal Component
    reversal_signals = pd.Series(index=data.index, dtype=float)
    
    # High Volatility: Trend consistency reversal
    intraday_return = (data['close'] - data['open']) / data['open']
    prev_day_momentum = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    high_vol_reversal = -np.sign(prev_day_momentum) * intraday_return
    
    # Low Volatility: Price-level reversal
    # Support/resistance: distance from 10-day high/low
    rolling_10d_high = data['high'].rolling(window=10, min_periods=5).max()
    rolling_10d_low = data['low'].rolling(window=10, min_periods=5).min()
    support_resistance = -((data['close'] - rolling_10d_low) / (rolling_10d_high - rolling_10d_low) - 0.5)
    
    # Round number effect: volume concentration at round price levels
    round_levels = np.round(data['close'] / 1.0) * 1.0  # Round to nearest dollar
    distance_to_round = np.abs(data['close'] - round_levels) / data['close']
    round_effect = 1.0 / (1.0 + distance_to_round * 100)  # Higher near round numbers
    
    # True Range
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - data['close'].shift(1))
    tr3 = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    low_vol_reversal = support_resistance * round_effect * true_range
    
    # Apply regime-specific reversal signals
    reversal_signals[high_vol_regime] = high_vol_reversal[high_vol_regime]
    reversal_signals[~high_vol_regime] = low_vol_reversal[~high_vol_regime]
    
    # Apply regime-dependent smoothing
    reversal_component = reversal_signals.copy()
    reversal_component[high_vol_regime] = reversal_signals[high_vol_regime].rolling(window=2, min_periods=1).mean()
    reversal_component[~high_vol_regime] = reversal_signals[~high_vol_regime].rolling(window=5, min_periods=3).mean()
    
    # Volume-Liquidity Efficiency Confirmation
    # Volume clustering efficiency
    volume_threshold = data['volume'].rolling(window=20, min_periods=10).quantile(0.7)
    high_volume_periods = data['volume'] > volume_threshold
    
    price_change = data['close'] - data['open']
    price_change_per_volume = price_change / data['volume']
    volume_clustering = price_change_per_volume[high_volume_periods].rolling(window=10, min_periods=5).sum()
    
    # Volume timing efficiency
    returns = (data['close'] - data['open']) / data['open']
    
    def rolling_corr(x, y, window):
        corrs = []
        for i in range(len(x)):
            if i >= window-1:
                start_idx = i - window + 1
                end_idx = i + 1
                corr = np.corrcoef(x[start_idx:end_idx], y[start_idx:end_idx])[0,1]
                corrs.append(corr if not np.isnan(corr) else 0)
            else:
                corrs.append(0)
        return pd.Series(corrs, index=x.index)
    
    current_correlation = rolling_corr(data['volume'], returns, 5)
    lagged_correlation = rolling_corr(data['volume'].shift(1), returns, 5)
    volume_timing = current_correlation - lagged_correlation
    
    # Liquidity efficiency
    volume_weighted_range = (data['high'] - data['low']) * data['volume']
    amount_efficiency = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    liquidity_multiplier = amount_efficiency / volume_weighted_range.replace(0, np.nan)
    
    # Volume-liquidity confirmation
    volume_liquidity_confirmation = volume_clustering * volume_timing * liquidity_multiplier
    
    # Volatility clustering adjustment
    range_expansion = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    volatility_persistence = data['close'].rolling(window=5, min_periods=3).std() / data['close'].rolling(window=10, min_periods=5).std()
    volatility_clustering_adjustment = range_expansion * volatility_persistence
    
    # Final Alpha Combination
    base_signal = breakout_component + reversal_component
    final_alpha = base_signal * volume_liquidity_confirmation * volatility_clustering_adjustment
    
    return final_alpha
