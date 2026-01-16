import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Compute Intraday Return
    intraday_return = ((df['high'] - df['low']) / df['open']) * np.sign(df['close'] - df['open'])
    
    # Calculate 5-day Rolling Volume Slope
    volume_slopes = []
    for t in range(len(df)):
        if t < 4:
            volume_slopes.append(0)
            continue
        X = np.arange(5).reshape(-1, 1)
        y = df['volume'].iloc[t-4:t+1].values
        model = LinearRegression().fit(X, y)
        volume_slopes.append(model.coef_[0])
    volume_slope = pd.Series(volume_slopes, index=df.index)
    
    # Adjust Momentum by Volume Slope
    momentum_volume = intraday_return * volume_slope
    
    # Calculate Volume Z-Score
    rolling_std = df['volume'].rolling(window=20, min_periods=1).std()
    z_score = df['volume'] / rolling_std
    
    # Combine Signals
    combined_signal = momentum_volume * z_score
    
    # Apply Min-Max Normalization
    normalized_signal = (combined_signal - combined_signal.min()) / (combined_signal.max() - combined_signal.min())
    
    return normalized_signal
