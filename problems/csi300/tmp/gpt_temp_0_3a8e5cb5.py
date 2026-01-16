import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day rolling slope using linear regression
    def rolling_slope(series):
        slope_values = []
        for i in range(len(series)):
            if i < 4:
                slope_values.append(np.nan)
            else:
                X = np.arange(5).reshape(-1, 1)
                y = series.iloc[i-4:i+1].values
                model = LinearRegression()
                model.fit(X, y)
                slope_values.append(model.coef_[0])
        return pd.Series(slope_values, index=series.index)

    # Calculate sign consistency
    def sign_consistency(slope_series):
        sign_count = []
        for i in range(len(slope_series)):
            if i < 4:
                sign_count.append(np.nan)
            else:
                signs = np.sign(slope_series.iloc[i-4:i+1])
                consistent_signs = np.sum(signs == np.sign(slope_series.iloc[i]))
                sign_count.append(consistent_signs / 5)
        return pd.Series(sign_count, index=slope_series.index)

    # Calculate normalized volume
    def normalized_volume(volume_series):
        return volume_series / volume_series.rolling(window=20, min_periods=1).mean()

    # Calculate Intraday Trend Persistence Factor
    slope = rolling_slope(df['close'])
    sign_consist = sign_consistency(slope)
    normalized_vol = normalized_volume(df['volume'])
    
    # Combine trend and volume confirmation
    factor = sign_consist * normalized_vol
    factor = factor.clip(lower=-3, upper=3)  # Cap extreme values
    
    return factor.dropna()
