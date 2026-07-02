import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 5-day Linear Regression Slope for Price and Volume
    def rolling_lr_slope(series, window):
        def lr_slope(x):
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(window=window).apply(lr_slope, raw=True)
    
    df['price_trend'] = rolling_lr_slope(df['close'], 5)
    df['volume_trend'] = rolling_lr_slope(df['volume'], 5)
    
    # Normalize Trends
    df['price_trend_norm'] = (df['price_trend'] / df['close'].shift(5)) * 100
    df['volume_trend_norm'] = (df['volume_trend'] / df['volume'].shift(5)) * 100
    
    # Signal Generation
    bullish_divergence = (df['price_trend_norm'] > 0) & (df['volume_trend_norm'] < 0)
    bearish_divergence = (df['price_trend_norm'] < 0) & (df['volume_trend_norm'] > 0)
    
    # Assign factor values
    factor = pd.Series(0, index=df.index)
    factor[bullish_divergence] = 1
    factor[bearish_divergence] = -1
    
    return factor
