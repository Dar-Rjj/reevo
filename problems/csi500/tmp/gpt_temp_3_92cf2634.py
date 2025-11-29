import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Gap Reversal Factor
    # Calculate Overnight Gap
    overnight_gap = (data['open'] / data['close'].shift(1) - 1)
    
    # Assess Intraday Reversal
    intraday_return = (data['close'] / data['open'] - 1)
    
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volume confirmation
    vol_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ratio = data['volume'] / vol_avg_5
    
    # Volatility-scaled signal
    gap_reversal = (overnight_gap * intraday_return) / true_range.replace(0, np.nan)
    factor1 = gap_reversal * volume_ratio
    
    # Range Expansion Momentum Divergence
    # Detect Range Expansion
    daily_range = data['high'] - data['low']
    range_avg_10 = daily_range.rolling(window=10, min_periods=1).mean()
    expansion_ratio = daily_range / range_avg_10.replace(0, np.nan)
    
    # Assess Momentum Divergence
    price_momentum = data['close'].pct_change(periods=3)
    volume_momentum = data['volume'].pct_change(periods=3)
    
    # Calculate divergence score
    divergence_score = np.sign(price_momentum) * np.sign(volume_momentum)
    
    # Magnitude weighting with ATR
    atr = true_range.rolling(window=14, min_periods=1).mean()
    volume_level = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    
    factor2 = divergence_score * expansion_ratio * atr * volume_level
    
    # Liquidity-Enhanced Breakout Detector
    # Identify Breakout Condition
    high_20 = data['high'].rolling(window=20, min_periods=1).max()
    low_20 = data['low'].rolling(window=20, min_periods=1).min()
    
    breakout_high = (data['high'] - high_20.shift(1)) / high_20.shift(1).replace(0, np.nan)
    breakout_low = (data['low'] - low_20.shift(1)) / low_20.shift(1).replace(0, np.nan)
    breakout_strength = breakout_high - breakout_low
    
    # Validate with Liquidity Signals
    vol_avg_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_deviation = data['volume'] / vol_avg_10.replace(0, np.nan)
    
    amount_avg_10 = data['amount'].rolling(window=10, min_periods=1).mean()
    amount_ratio = data['amount'] / amount_avg_10.replace(0, np.nan)
    
    # Trend filter
    price_momentum_5 = data['close'].pct_change(periods=5)
    direction_consistency = price_momentum_5.rolling(window=3, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(1, len(x)-1)
    )
    
    factor3 = breakout_strength * volume_deviation * amount_ratio * direction_consistency
    
    # Gap-Fill Efficiency Momentum
    # Measure Gap Characteristics
    opening_gap = (data['open'] / data['close'].shift(1) - 1)
    
    # Assess Fill Completion
    gap_fill = np.where(opening_gap > 0,
                       (data['low'] - data['open']) / (data['close'].shift(1) - data['open']).replace(0, np.nan),
                       (data['high'] - data['open']) / (data['close'].shift(1) - data['open']).replace(0, np.nan))
    
    # Calculate Intraday Momentum
    normalized_range = daily_range / data['close'].rolling(window=5, min_periods=1).std().replace(0, np.nan)
    
    # Volume timing
    volume_shift = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    
    factor4 = gap_fill * normalized_range * volume_shift
    
    # Price-Volume Convergence Oscillator
    # Calculate Short-term Trends
    def linear_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    price_trend = data['close'].rolling(window=5, min_periods=1).apply(linear_slope, raw=False)
    volume_trend = data['volume'].rolling(window=5, min_periods=1).apply(linear_slope, raw=False)
    
    # Assess Convergence Patterns
    convergence_score = np.sign(price_trend) * np.sign(volume_trend)
    
    # Apply Volatility Context
    range_ratio = daily_range / range_avg_10.replace(0, np.nan)
    
    factor5 = convergence_score * range_ratio
    
    # Intraday Pressure Decay Factor
    # Calculate Pressure Components
    buying_pressure = ((data['close'] - data['open']) / data['open'].replace(0, np.nan) + 
                      (data['close'] - data['low']) / data['low'].replace(0, np.nan)) / 2
    
    selling_pressure = ((data['open'] - data['close']) / data['open'].replace(0, np.nan) + 
                       (data['high'] - data['close']) / data['high'].replace(0, np.nan)) / 2
    
    # Calculate Net Pressure
    net_pressure = (buying_pressure - selling_pressure) / daily_range.replace(0, np.nan)
    
    # Weight by Volume-Amount Ratio
    volume_amount_ratio = data['volume'] / data['amount'].replace(0, np.nan)
    prev_ratio = volume_amount_ratio.shift(1)
    prev_pressure = net_pressure.shift(1)
    
    factor6 = net_pressure * volume_amount_ratio * prev_ratio.fillna(1) * prev_pressure.fillna(0)
    
    # Volatility Clustering Reversal
    # Identify Volatility Patterns
    daily_returns = data['close'].pct_change()
    
    def clustering_intensity(returns):
        if len(returns) < 4:
            return 0
        current = returns.iloc[-1]
        prev_3 = returns.iloc[-4:-1]
        return np.mean([abs(current - prev) for prev in prev_3])
    
    clustering = daily_returns.rolling(window=4, min_periods=1).apply(clustering_intensity, raw=False)
    
    # Generate Reversal Signal
    reversal_magnitude = -clustering * np.sign(daily_returns)
    
    # Volume confirmation
    high_vol_threshold = daily_returns.rolling(window=20, min_periods=1).std()
    is_high_vol = abs(daily_returns) > high_vol_threshold
    vol_high = data['volume'].where(is_high_vol).rolling(window=10, min_periods=1).mean()
    vol_normal = data['volume'].where(~is_high_vol).rolling(window=10, min_periods=1).mean()
    volume_ratio_vol = vol_high / vol_normal.replace(0, np.nan)
    
    # Range scaling
    atr_5 = true_range.rolling(window=5, min_periods=1).mean()
    
    factor7 = reversal_magnitude * volume_ratio_vol * true_range / atr_5.replace(0, np.nan)
    
    # Momentum Persistence with Liquidity
    # Assess Momentum Quality
    momentum_3 = data['close'].pct_change(periods=3)
    
    def consistency_score(returns):
        if len(returns) < 2:
            return 0
        return len([i for i in range(1, len(returns)) if np.sign(returns.iloc[i]) == np.sign(returns.iloc[i-1])]) / (len(returns)-1)
    
    momentum_consistency = momentum_3.rolling(window=5, min_periods=1).apply(consistency_score, raw=False)
    
    # Evaluate Duration
    def persistence_factor(returns):
        if len(returns) < 2:
            return 1
        current_sign = np.sign(returns.iloc[-1])
        streak = 0
        for i in range(len(returns)-1, -1, -1):
            if np.sign(returns.iloc[i]) == current_sign:
                streak += 1
            else:
                break
        return streak
    
    persistence = momentum_3.rolling(window=10, min_periods=1).apply(persistence_factor, raw=False)
    
    # Validate with Liquidity Metrics
    volume_trend_5 = data['volume'].rolling(window=5, min_periods=1).apply(linear_slope, raw=False)
    volume_stability = 1 / (data['volume'].rolling(window=5, min_periods=1).std() / data['volume'].rolling(window=5, min_periods=1).mean()).replace(0, np.nan)
    
    amount_trend_5 = data['amount'].rolling(window=5, min_periods=1).apply(linear_slope, raw=False)
    amount_consistency = np.sign(volume_trend_5) * np.sign(amount_trend_5)
    
    liquidity_score = volume_stability * amount_consistency
    
    factor8 = momentum_3 * persistence * liquidity_score
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'f1': factor1,
        'f2': factor2,
        'f3': factor3,
        'f4': factor4,
        'f5': factor5,
        'f6': factor6,
        'f7': factor7,
        'f8': factor8
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20, min_periods=1).mean()) / x.rolling(window=20, min_periods=1).std().replace(0, np.nan))
    
    # Equal weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
