import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Price Trend Component
    def linear_regression_slope(series, window):
        slopes = pd.Series(np.nan, index=series.index)
        for i in range(window - 1, len(series)):
            x = np.arange(window).reshape(-1, 1)
            y = series.iloc[i - window + 1:i + 1].values
            model = LinearRegression().fit(x, y)
            slopes.iloc[i] = model.coef_[0]
        return slopes

    short_term_price_trend = linear_regression_slope(df['close'], 5)
    medium_term_price_trend = linear_regression_slope(df['close'], 20)
    
    # Volume Trend Component
    def ema_slope(series, window):
        ema = series.ewm(span=window, adjust=False).mean()
        slope = ema.diff()
        return slope

    short_term_volume_trend = ema_slope(df['volume'], 5)
    medium_term_volume_trend = ema_slope(df['volume'], 20)
    
    # Divergence Signal
    def divergence_signal(price_trend, volume_trend):
        signal = pd.Series(np.nan, index=price_trend.index)
        signal[(price_trend > 0) & (volume_trend < 0)] = -1
        signal[(price_trend < 0) & (volume_trend > 0)] = 1
        return signal.fillna(0)

    divergence = divergence_signal(short_term_price_trend, short_term_volume_trend)
    
    # Magnitude Adjustment
    price_volatility = df['close'].rolling(window=20).std()
    historical_volatility = price_volatility.rolling(window=252).mean()
    volatility_adjustment = price_volatility / historical_volatility
    
    volume_trend_std = medium_term_volume_trend.rolling(window=20).std()
    
    factor = divergence * volatility_adjustment / volume_trend_std
    return factor
