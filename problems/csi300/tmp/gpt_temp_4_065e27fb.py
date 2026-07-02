import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price acceleration (second derivative)
    price_acceleration = df['close'].diff().diff()
    
    # Volume volatility (standard deviation)
    volume_volatility = df['volume'].rolling(5).std()
    
    # Combined factor with geometric mean smoothing
    combined_factor = (price_acceleration * volume_volatility).abs()
    heuristics_matrix = combined_factor.rolling(7, min_periods=1).apply(lambda x: np.prod(x)**(1/len(x)))
    
    return heuristics_matrix
