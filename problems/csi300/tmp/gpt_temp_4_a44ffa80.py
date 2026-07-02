import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Price Trend Component
    def calculate_trend(series, window):
        slopes = np.zeros(len(series))
        for i in range(window, len(series)):
            X = np.arange(window).reshape(-1, 1)
            y = series[i-window:i].values
            model = LinearRegression().fit(X, y)
            slopes[i] = model.coef_[0]
        return slopes

    short_term_trend = calculate_trend(df['close'], 5)
    long_term_trend = calculate_trend(df['close'], 20)

    # Volume Trend Component
    volume_trend = calculate_trend(df['volume'], 5)
    volume_std = df['volume'].rolling(20, min_periods=1).std()
    normalized_volume_trend = volume_trend / volume_std
    z_score = (normalized_volume_trend - normalized_volume_trend.rolling(20, min_periods=1).mean()) / normalized_volume_trend.rolling(20, min_periods=1).std()

    # Divergence Signal
    divergence_signal = np.zeros(len(df))
    for i in range(1, len(df)):
        if short_term_trend[i] > 0 and z_score[i] < 0:
            divergence_signal[i] = -1
        elif short_term_trend[i] < 0 and z_score[i] > 0:
            divergence_signal[i] = 1

    # Weight by Trend Strength
    trend_strength = np.abs(short_term_trend)
    factor_values = divergence_signal * trend_strength
    factor_values = 1 / (1 + np.exp(-factor_values))  # Logistic transformation

    return pd.Series(factor_values, index=df.index)
