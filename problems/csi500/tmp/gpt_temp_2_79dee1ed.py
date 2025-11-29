import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel cross-sectional alpha factors using range expansion, intraday efficiency,
    multi-timeframe convergence, amount impact, and fractal analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Range Expansion Momentum Divergence
    # Calculate abnormal range expansion ratio
    daily_range = (data['high'] - data['low']) / data['close']
    range_ma_5 = daily_range.rolling(window=5, min_periods=3).mean()
    expansion_ratio = daily_range / range_ma_5
    
    # Volume-momentum divergence
    price_momentum_10 = data['close'].pct_change(periods=10)
    volume_trend_10 = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan
    )
    momentum_volume_divergence = price_momentum_10 * (price_momentum_10 - volume_trend_10)
    
    factor1 = expansion_ratio * momentum_volume_divergence
    
    # Factor 2: Intraday Path Efficiency with Volume Confirmation
    # Calculate price movement efficiency ratio
    optimal_path = np.abs(data['high'] - data['low'])
    actual_movement = np.abs(data['close'] - data['open'])
    efficiency_ratio = optimal_path / (actual_movement + 1e-8)
    
    # Volume regime adjustment
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=10).mean()
    high_volume_regime = (data['volume'] > volume_ma_20).astype(int)
    low_volume_regime = (data['volume'] <= volume_ma_20).astype(int)
    
    high_vol_efficiency = efficiency_ratio * high_volume_regime
    low_vol_efficiency = efficiency_ratio * low_volume_regime
    
    efficiency_diff = high_vol_efficiency.rolling(window=10, min_periods=5).mean() - \
                     low_vol_efficiency.rolling(window=10, min_periods=5).mean()
    
    factor2 = efficiency_ratio * efficiency_diff
    
    # Factor 3: Multi-Timeframe Convergence with Reversal Detection
    # Volatility-momentum convergence patterns
    momentum_3 = data['close'].pct_change(periods=3)
    momentum_10 = data['close'].pct_change(periods=10)
    
    volatility_1 = data['close'].pct_change().abs()
    volatility_5 = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    
    momentum_convergence = momentum_3 * momentum_10
    volatility_convergence = volatility_1 * volatility_5
    
    # Intraday reversal confirmation
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_counter = (data['close'] - data['open']) / data['open']
    reversal_strength = np.abs(overnight_gap * intraday_counter)
    
    factor3 = momentum_convergence * volatility_convergence * reversal_strength
    
    # Factor 4: Amount-Based Impact with Persistence Tracking
    # Price impact per unit amount
    price_change = data['close'].pct_change()
    amount_impact = price_change / (data['amount'] + 1e-8)
    
    # Impact persistence patterns
    impact_direction = np.sign(amount_impact)
    consecutive_direction = impact_direction.rolling(window=5, min_periods=3).apply(
        lambda x: len(set(x)) if len(x) >= 3 else np.nan
    )
    
    impact_baseline = amount_impact.rolling(window=20, min_periods=10).mean()
    regime_shift = amount_impact - impact_baseline
    
    factor4 = amount_impact * consecutive_direction * regime_shift
    
    # Factor 5: Fractal Market Structure Analysis
    # Price movement fractal dimension approximation
    def hurst_exponent(series, max_lag=10):
        if len(series) < max_lag:
            return np.nan
        lags = range(2, max_lag + 1)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    price_fractal = data['close'].rolling(window=30, min_periods=20).apply(
        lambda x: hurst_exponent(x) if len(x) >= 20 else np.nan
    )
    
    # Volume pattern fractal dimension
    volume_fractal = data['volume'].rolling(window=30, min_periods=20).apply(
        lambda x: hurst_exponent(x) if len(x) >= 20 else np.nan
    )
    
    # Price-volume fractal correlation
    pv_correlation = data['close'].pct_change().rolling(window=20, min_periods=10).corr(
        data['volume'].pct_change()
    )
    
    factor5 = price_fractal * volume_fractal * pv_correlation
    
    # Combine factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Final combined factor (equal weighted)
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
