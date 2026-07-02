import pandas as pd
import numpy as np
def heuristics_v2(df):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Calculate 5-day Price Slope
    def calculate_price_slope(close):
        slopes = []
        for i in range(len(close)):
            if i < 4:
                slopes.append(np.nan)
            else:
                x = np.arange(5).reshape(-1, 1)
                y = close[i-4:i+1].values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x, y)
                slopes.append(model.coef_[0][0])
        return pd.Series(slopes, index=close.index)

    # Calculate 5-day Volume Slope
    def calculate_volume_slope(volume):
        slopes = []
        for i in range(len(volume)):
            if i < 4:
                slopes.append(np.nan)
            else:
                x = np.arange(5).reshape(-1, 1)
                y = volume[i-4:i+1].values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x, y)
                slopes.append(model.coef_[0][0])
        return pd.Series(slopes, index=volume.index)

    # Price Trend Component
    price_slope = calculate_price_slope(df['close'])

    # Volume Trend Component
    volume_slope = calculate_volume_slope(df['volume'])

    # Detect Divergence
    divergence_signal = np.sign(price_slope) * np.sign(volume_slope)
    divergence_magnitude = np.abs(price_slope) * np.abs(volume_slope)
    divergence_factor = divergence_signal * divergence_magnitude

    return divergence_factor
