import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day Price Momentum
    df['momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Normalize by 20-day rolling volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    df['normalized_momentum'] = df['momentum'] / df['volatility']
    
    # Calculate 5-day Volume Slope using linear regression
    volume_slopes = []
    for i in range(len(df)):
        if i < 4:
            volume_slopes.append(np.nan)
        else:
            X = np.arange(5).reshape(-1, 1)
            y = df['volume'].iloc[i-4:i+1].values
            model = LinearRegression().fit(X, y)
            volume_slopes.append(model.coef_[0])
    df['volume_slope'] = volume_slopes
    
    # Calculate 5-day rolling correlation between Price Momentum and Volume Slope
    df['correlation'] = df['normalized_momentum'].rolling(window=5).corr(df['volume_slope'])
    
    # Generate Divergence Signal
    df['divergence_signal'] = np.where(df['correlation'] < -0.5, -1 * abs(df['normalized_momentum']), 0)
    
    return df['divergence_signal']
