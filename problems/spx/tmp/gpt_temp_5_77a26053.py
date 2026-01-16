import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['open']
    
    # Price Momentum
    df['price_momentum'] = df['close'].pct_change(periods=10)
    
    # Combined Momentum
    combined_momentum = intraday_momentum + df['price_momentum']
    
    # Liquidity Adjustment: Volume Slope
    def calculate_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    volume_slope = df['volume'].rolling(window=5).apply(calculate_slope, raw=False)
    
    # Factor Calculation: Multiply Combined Momentum by Volume Slope
    factor = combined_momentum * volume_slope
    
    return factor
