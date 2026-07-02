import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Short-Term Price Trend (5-day slope)
    def calc_price_slope_short(df):
        slopes = []
        for i in range(len(df)):
            if i < 5:
                slopes.append(np.nan)
            else:
                X = np.arange(5).reshape(-1, 1)
                y = df['close'].iloc[i-5:i].values
                model = LinearRegression().fit(X, y)
                slopes.append(model.coef_[0])
        return pd.Series(slopes, index=df.index)
    
    # Calculate Medium-Term Price Trend (20-day slope)
    def calc_price_slope_medium(df):
        slopes = []
        for i in range(len(df)):
            if i < 20:
                slopes.append(np.nan)
            else:
                X = np.arange(20).reshape(-1, 1)
                y = df['close'].iloc[i-20:i].values
                model = LinearRegression().fit(X, y)
                slopes.append(model.coef_[0])
        return pd.Series(slopes, index=df.index)
    
    # Calculate Short-Term Volume Trend (5-day slope)
    def calc_volume_slope_short(df):
        slopes = []
        for i in range(len(df)):
            if i < 5:
                slopes.append(np.nan)
            else:
                X = np.arange(5).reshape(-1, 1)
                y = df['volume'].iloc[i-5:i].values
                model = LinearRegression().fit(X, y)
                slopes.append(model.coef_[0])
        return pd.Series(slopes, index=df.index)
    
    # Calculate Medium-Term Volume Trend (20-day slope)
    def calc_volume_slope_medium(df):
        slopes = []
        for i in range(len(df)):
            if i < 20:
                slopes.append(np.nan)
            else:
                X = np.arange(20).reshape(-1, 1)
                y = df['volume'].iloc[i-20:i].values
                model = LinearRegression().fit(X, y)
                slopes.append(model.coef_[0])
        return pd.Series(slopes, index=df.index)
    
    # Calculate Divergence Signal
    def calc_divergence_signal(price_slope, volume_slope):
        signal = np.where((price_slope > 0) & (volume_slope < 0), -1,
                         np.where((price_slope < 0) & (volume_slope > 0), 1, 0))
        return pd.Series(signal, index=price_slope.index)
    
    # Main Calculation
    price_slope_short = calc_price_slope_short(df)
    price_slope_medium = calc_price_slope_medium(df)
    volume_slope_short = calc_volume_slope_short(df)
    volume_slope_medium = calc_volume_slope_medium(df)
    
    divergence_short = calc_divergence_signal(price_slope_short, volume_slope_short)
    divergence_medium = calc_divergence_signal(price_slope_medium, volume_slope_medium)
    
    # Combine signals with magnitude adjustment
    factor_short = divergence_short * np.abs(price_slope_short) * np.abs(volume_slope_short)
    factor_medium = divergence_medium * np.abs(price_slope_medium) * np.abs(volume_slope_medium)
    
    # Final factor is the average of short and medium term
    factor = (factor_short + factor_medium) / 2
    
    return factor.dropna()
