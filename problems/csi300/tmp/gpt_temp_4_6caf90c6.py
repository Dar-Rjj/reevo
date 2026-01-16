import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day price slope using linear regression
    def calculate_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope

    # Calculate the 5-day slope for the close prices
    df['price_slope'] = df['close'].rolling(window=5, min_periods=5).apply(calculate_slope, raw=True)

    # Classify trend as Up (1) or Down (-1)
    df['trend'] = np.where(df['price_slope'] > 0, 1, -1)

    # Compute volume Z-score
    df['volume_mean'] = df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20, min_periods=20).std()
    df['volume_z'] = (df['volume'] - df['volume_mean']) / df['volume_std']

    # Calculate normalized price change
    df['price_change'] = (df['close'] - df['open']) / (df['high'] - df['low'])

    # Generate signal
    df['signal'] = df['trend'] * df['volume_z']

    # Handle cases where volume_z is NaN (due to insufficient data)
    df['signal'] = df['signal'].fillna(0)

    return df['signal']
