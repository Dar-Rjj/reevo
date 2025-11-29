import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['returns'] = (data['close'] - data['prev_close']) / data['prev_close']
    
    # Intraday Momentum Efficiency
    data['intraday_range'] = (data['high'] - data['low']) / data['low']
    data['price_direction'] = np.sign(data['close'] - data['prev_close'])
    data['range_direction_product'] = data['intraday_range'] * data['price_direction']
    
    data['volume_ratio'] = data['volume'] / data['prev_volume']
    data['volume_adjusted_momentum'] = data['range_direction_product'] * data['volume_ratio']
    
    data['gap_size'] = np.abs(data['open'] - data['prev_close']) / data['prev_close']
    data['gap_persistence'] = data['gap_size'].rolling(window=5, min_periods=3).mean()
    data['ime_factor'] = data['volume_adjusted_momentum'] / (1 + data['gap_persistence'])
    
    # Relative Strength Efficiency Composite
    # Market return approximation (using close as proxy)
    data['market_return'] = data['close'].pct_change()
    data['relative_strength'] = data['returns'] - data['market_return']
    
    data['abs_price_change'] = np.abs(data['close'] - data['prev_close'])
    data['high_low_range'] = data['high'] - data['low']
    data['efficiency_ratio'] = np.where(data['high_low_range'] > 0, 
                                       data['abs_price_change'] / data['high_low_range'], 0)
    
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
    
    data['efficiency_strength'] = data['efficiency_ratio'] * data['relative_strength'] * data['volume_ratio_20']
    
    # Exponential weighting
    decay_factor = 0.94
    weights = np.array([decay_factor ** i for i in range(10)][::-1])
    weights = weights / weights.sum()
    
    data['rsec_factor'] = data['efficiency_strength'].rolling(window=10, min_periods=5).apply(
        lambda x: np.dot(x, weights) if len(x) == 10 else np.nan, raw=True
    )
    
    # Volatility-Regime Adaptive Momentum
    data['realized_vol_20'] = data['returns'].rolling(window=20, min_periods=10).std()
    vol_median = data['realized_vol_20'].median()
    data['vol_regime'] = np.where(data['realized_vol_20'] > vol_median, 'high', 'low')
    
    data['momentum_2d'] = data['close'].pct_change(periods=2)
    data['momentum_10d'] = data['close'].pct_change(periods=10)
    
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma_5']
    
    data['adaptive_momentum'] = np.where(
        data['vol_regime'] == 'high',
        data['momentum_2d'] * data['volume_trend'],
        data['momentum_10d'] * data['volume_trend']
    )
    
    # Fractal Price-Volume Congruence
    def hurst_exponent(series, window=15):
        """Calculate Hurst exponent as fractal dimension proxy"""
        if len(series) < window:
            return np.nan
        
        lags = range(2, min(8, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['price_fractal'] = data['close'].rolling(window=15, min_periods=10).apply(
        lambda x: hurst_exponent(x.values), raw=False
    )
    data['volume_fractal'] = data['volume'].rolling(window=15, min_periods=10).apply(
        lambda x: hurst_exponent(x.values), raw=False
    )
    
    data['fractal_correlation'] = data['price_fractal'].rolling(window=10, min_periods=5).corr(data['volume_fractal'])
    
    data['trend_3d'] = data['close'].pct_change(periods=3)
    data['trend_10d'] = data['close'].pct_change(periods=10)
    
    def cosine_similarity(a, b):
        if np.isnan(a) or np.isnan(b) or a == 0 or b == 0:
            return 0
        return (a * b) / (np.sqrt(a**2 + b**2) + 1e-8)
    
    data['trend_congruence'] = data.apply(
        lambda row: cosine_similarity(row['trend_3d'], row['trend_10d']), axis=1
    )
    
    # Support/Resistance approximation using recent highs/lows
    data['recent_high'] = data['high'].rolling(window=10, min_periods=5).max()
    data['recent_low'] = data['low'].rolling(window=10, min_periods=5).min()
    data['break_magnitude'] = np.where(
        data['close'] > data['recent_high'],
        (data['close'] - data['recent_high']) / data['recent_high'],
        np.where(data['close'] < data['recent_low'],
                (data['close'] - data['recent_low']) / data['recent_low'], 0)
    )
    
    data['fractal_factor'] = (data['fractal_correlation'] * data['trend_congruence'] * 
                             data['break_magnitude'] * data['volume_ratio_20'])
    
    # Combine all factors with equal weighting
    factors = ['ime_factor', 'rsec_factor', 'adaptive_momentum', 'fractal_factor']
    valid_factors = [data[factor] for factor in factors if factor in data.columns]
    
    if valid_factors:
        # Z-score normalization for each factor
        z_scores = []
        for factor in valid_factors:
            mean_val = factor.rolling(window=20, min_periods=10).mean()
            std_val = factor.rolling(window=20, min_periods=10).std()
            z_score = (factor - mean_val) / (std_val + 1e-8)
            z_scores.append(z_score)
        
        # Equal weighted combination
        combined_factor = sum(z_scores) / len(z_scores)
    else:
        combined_factor = pd.Series(index=data.index, dtype=float)
    
    return combined_factor
