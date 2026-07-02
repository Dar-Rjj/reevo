import pandas as pd
import numpy as np
def heuristics_v2(df):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Calculate first derivative of price
    df['FirstDeriv'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate second derivative of price
    df['SecondDeriv'] = (df['FirstDeriv'] - df['FirstDeriv'].shift(1)) / df['FirstDeriv'].shift(1)
    
    # Normalize by volatility (rolling 5-day std dev of returns)
    df['Returns'] = df['close'].pct_change()
    df['StdDev'] = df['Returns'].rolling(window=5, min_periods=1).std()
    df['NormalizedAcceleration'] = df['SecondDeriv'] / df['StdDev']
    
    # Calculate volume trend (rolling 3-day volume slope)
    volume_slopes = []
    for i in range(len(df)):
        if i >= 2:
            X = np.arange(3).reshape(-1, 1)
            y = df['volume'].iloc[i-2:i+1].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
        else:
            slope = 0
        volume_slopes.append(slope)
    df['VolumeTrend'] = volume_slopes
    
    # Combine components
    df['Factor'] = df['NormalizedAcceleration'] * df['VolumeTrend']
    
    # Apply exponential smoothing with factor 0.5
    df['SmoothedFactor'] = df['Factor'].ewm(alpha=0.5, adjust=False).mean()
    
    return df['SmoothedFactor']
