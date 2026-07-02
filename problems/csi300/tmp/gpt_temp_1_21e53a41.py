import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Price Trend Component
    def calculate_slope(series, window):
        slopes = []
        for i in range(window, len(series)):
            X = np.arange(window).reshape(-1, 1)
            y = series[i-window:i].values
            model = LinearRegression().fit(X, y)
            slopes.append(model.coef_[0])
        return slopes

    close_prices = data['close']
    price_slope = calculate_slope(close_prices, 5)
    price_slope = [0] * 5 + price_slope  # Pad with zeros for the first 5 days

    high_low_range = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    normalized_price_slope = np.array(price_slope) / high_low_range.replace(0, 1)

    # Volume Divergence Component
    volume = data['volume']
    volume_slope = calculate_slope(volume, 5)
    volume_slope = [0] * 5 + volume_slope  # Pad with zeros for the first 5 days

    # Compare Volume Slope with Price Slope
    divergence_factor = np.array(normalized_price_slope) * np.sign(np.array(volume_slope) - np.array(price_slope))

    return pd.Series(divergence_factor, index=data.index)
