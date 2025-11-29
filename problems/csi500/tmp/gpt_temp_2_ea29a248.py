import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate composite alpha factors using price, volume, amount and range data
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['open_close_return'] = (data['close'] - data['open']) / data['open']
    data['volume_amount_ratio'] = data['volume'] / (data['amount'] + 1e-8)
    
    # Calculate rolling averages
    data['range_5d_avg'] = data['high_low_range'].rolling(window=5).mean()
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['range_10d_avg'] = data['high_low_range'].rolling(window=10).mean()
    
    # Factor 1: High-Low Range Momentum Decay Factor
    range_momentum = (data['high_low_range'] - data['high_low_range'].shift(5)) / (data['high_low_range'].shift(5) + 1e-8)
    volume_amount_ratio_change = data['volume_amount_ratio'] / (data['volume_amount_ratio'].shift(1) + 1e-8)
    factor1 = range_momentum * volume_amount_ratio_change
    
    # Factor 2: Intraday Gap Volatility Factor
    gap_percentage = abs((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8))
    range_momentum_daily = data['high_low_range'] / (data['high_low_range'].shift(1) + 1e-8)
    volume_surge = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    factor2 = gap_percentage / (range_momentum_daily + 1e-8) * np.minimum(volume_surge, 3)
    
    # Factor 3: Amount-Weighted Range Divergence
    price_momentum = data['close'].pct_change(5)
    range_momentum_5d = data['high_low_range'].pct_change(5)
    divergence = np.where(
        (price_momentum > 0) & (range_momentum_5d < 0), -1,
        np.where((price_momentum < 0) & (range_momentum_5d > 0), 1, 0)
    )
    amount_ratio = data['amount'] / (data['amount'].shift(1) + 1e-8)
    factor3 = divergence * amount_ratio * data['open_close_return'] * data['high_low_range']
    
    # Factor 4: Volume-Pressure Range Expansion
    range_expansion = data['high_low_range'] / (data['range_10d_avg'] + 1e-8)
    volume_pressure = data['volume'] / (data['volume_10d_avg'] + 1e-8)
    price_momentum_3d = data['close'].pct_change(3)
    range_momentum_dir = np.sign(data['high_low_range'] - data['high_low_range'].shift(1))
    factor4 = range_expansion * volume_pressure * price_momentum_3d * range_momentum_dir
    
    # Factor 5: Gap-Fill Momentum Efficiency
    gap_pct = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    gap_fill_efficiency = 1 - abs(data['close'] - data['open']) / (abs(gap_pct * data['close'].shift(1)) + 1e-8)
    amount_stability = data['amount'] / (data['amount'].shift(1) + 1e-8)
    factor5 = gap_pct * gap_fill_efficiency * data['volume'] * amount_stability
    
    # Factor 6: Breakout Volume-Momentum Validation
    high_5d = data['high'].rolling(window=5).max()
    low_5d = data['low'].rolling(window=5).min()
    breakout_strength = np.where(
        data['close'] > high_5d.shift(1), (data['close'] - high_5d.shift(1)) / high_5d.shift(1),
        np.where(data['close'] < low_5d.shift(1), (low_5d.shift(1) - data['close']) / low_5d.shift(1), 0)
    )
    volume_surge_5d = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    range_factor = data['high_low_range'] / (data['range_5d_avg'] + 1e-8)
    factor6 = breakout_strength * volume_surge_5d * range_factor
    
    # Factor 7: Price-Volume-Range Convergence
    def linear_trend(series, window=5):
        x = np.arange(window)
        trends = []
        for i in range(len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    trends.append(slope)
                else:
                    trends.append(np.nan)
            else:
                trends.append(np.nan)
        return pd.Series(trends, index=series.index)
    
    price_trend = linear_trend(data['close'])
    volume_trend = linear_trend(data['volume'])
    range_trend = linear_trend(data['high_low_range'])
    
    convergence_score = np.sign(price_trend) * np.sign(volume_trend) * np.sign(range_trend)
    magnitude_weight = data['volume_10d_avg'] * data['range_10d_avg'] * data['close']
    factor7 = convergence_score * magnitude_weight
    
    # Factor 8: Momentum Decay with Range Support
    momentum_3d = data['close'].pct_change(3)
    momentum_5d = data['close'].pct_change(5)
    momentum_consistency = np.sign(momentum_3d) * np.sign(momentum_5d)
    range_trend_5d = linear_trend(data['high_low_range'])
    volume_amount_stability = (data['volume'] / data['volume_5d_avg']) * (data['amount'] / data['amount'].rolling(5).mean())
    factor8 = momentum_consistency * range_trend_5d * volume_amount_stability
    
    # Combine all factors with equal weighting
    factors = [factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8]
    valid_factors = []
    
    for factor in factors:
        if len(factor.dropna()) > 0:
            # Normalize each factor
            normalized_factor = (factor - factor.mean()) / (factor.std() + 1e-8)
            valid_factors.append(normalized_factor)
    
    if valid_factors:
        # Equal weighted combination
        combined_factor = sum(valid_factors) / len(valid_factors)
        result = combined_factor
    
    return result
