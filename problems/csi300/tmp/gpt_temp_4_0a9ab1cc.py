import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Slope
    def calc_price_slope(series):
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope

    price_slope = df['close'].rolling(window=5, min_periods=5).apply(calc_price_slope, raw=True)

    # Calculate 5-day Volume Slope
    def calc_volume_slope(series):
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope

    volume_slope = df['volume'].rolling(window=5, min_periods=5).apply(calc_volume_slope, raw=True)

    # Generate Divergence Signal
    def divergence_signal(price, volume):
        if price > 0 and volume < 0:
            return -1
        elif price < 0 and volume > 0:
            return 1
        else:
            return 0

    factor = pd.Series(np.vectorize(divergence_signal)(price_slope, volume_slope), index=df.index)
    
    return factor
