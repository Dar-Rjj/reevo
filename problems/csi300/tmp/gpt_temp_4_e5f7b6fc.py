import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate Price Momentum
    momentum_window = 10
    data['momentum'] = data['close'].rolling(window=momentum_window, min_periods=1).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)

    # Detect Volume Spike
    volume_window = 5
    data['volume_growth_rate'] = data['volume'].pct_change(periods=volume_window)
    spike_threshold = data['volume_growth_rate'].quantile(0.75)  # 75th percentile as threshold
    data['volume_spike'] = data['volume_growth_rate'] > spike_threshold

    # Calculate Price Range
    range_window = 10
    data['rolling_high'] = data['high'].rolling(window=range_window, min_periods=1).max()
    data['rolling_low'] = data['low'].rolling(window=range_window, min_periods=1).min()
    data['price_range'] = data['rolling_high'] - data['rolling_low']

    # Calculate Volatility
    volatility_window = 10
    data['volatility'] = data['close'].rolling(window=volatility_window, min_periods=1).std()

    # Combine Momentum, Spike, Range, and Volatility
    narrow_range_threshold = data['price_range'].quantile(0.25)  # 25th percentile as narrow range threshold
    low_volatility_threshold = data['volatility'].quantile(0.25)  # 25th percentile as low volatility threshold

    data['factor'] = data.apply(lambda row: 
        -row['momentum'] if (
            row['volume_spike'] and 
            row['price_range'] < narrow_range_threshold and 
            row['volatility'] < low_volatility_threshold
        ) else row['momentum'], axis=1)

    return data['factor']
