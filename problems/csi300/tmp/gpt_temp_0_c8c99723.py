import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    # Rolling 5-Day Price Slope
    price_slope = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)
    price_slope_normalized = price_slope / df['close']
    
    # Price Direction Consistency
    price_direction = df['close'].diff()
    positive_days = price_direction.rolling(window=5).apply(lambda x: (x > 0).sum(), raw=True)
    price_direction_consistency = positive_days / 5
    
    # Volume Divergence Component
    # Volume Spike Detection
    volume_mean = df['volume'].rolling(window=10).mean()
    volume_std = df['volume'].rolling(window=10).std()
    volume_z_score = (df['volume'] - volume_mean) / volume_std
    volume_z_score_abs = volume_z_score.abs()
    
    # Volume-Price Correlation
    volume_price_corr = df['volume'].rolling(window=5).corr(df['close']) * -1
    volume_price_corr_weighted = volume_price_corr * volume_z_score_abs
    
    # Combine components to form the factor
    factor = price_slope_normalized * price_direction_consistency * volume_price_corr_weighted
    
    return factor
