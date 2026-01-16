import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_momentum'] = df['close'] / df['open'] - 1

    # Calculate Breakout Ratio
    df['rolling_max_high'] = df['high'].rolling(window=5, min_periods=1).max()
    df['breakout_ratio'] = df['high'] / df['rolling_max_high'] - 1

    # Measure Momentum Breakout
    df['momentum_breakout'] = df['intraday_momentum'] * df['breakout_ratio']

    # Calculate Volume Trend
    df['avg_volume_10d'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume_10d']

    # Apply Linear Regression Slope (5d) to Volume Ratio
    def calculate_slope(x):
        model = LinearRegression()
        X = np.arange(len(x)).reshape(-1, 1)
        model.fit(X, x)
        return model.coef_[0]

    df['volume_trend_slope'] = df['volume_ratio'].rolling(window=5, min_periods=1).apply(calculate_slope, raw=False)

    # Adjust Momentum Breakout by Volume Trend
    df['adjusted_momentum_breakout'] = df['momentum_breakout'] * df['volume_trend_slope']

    # Normalize Factor Value
    median = df['adjusted_momentum_breakout'].rolling(window=252, min_periods=1).median()
    iqr = df['adjusted_momentum_breakout'].rolling(window=252, min_periods=1).quantile(0.75) - df['adjusted_momentum_breakout'].rolling(window=252, min_periods=1).quantile(0.25)
    df['factor_value'] = (df['adjusted_momentum_breakout'] - median) / iqr

    return df['factor_value']
