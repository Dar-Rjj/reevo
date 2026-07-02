import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    def calculate_trend(series, window=5):
        trends = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            x = np.arange(window)
            y = series.iloc[i-window+1:i+1].values
            slope, _, _, _, _ = linregress(x, y)
            trends.iloc[i] = slope
        return trends

    price_trend = calculate_trend(df['close'])
    volume_trend = calculate_trend(df['volume'])
    
    divergence = price_trend * volume_trend
    factor = np.sign(divergence)
    
    return factor
