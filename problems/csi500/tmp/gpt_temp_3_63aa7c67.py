import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate intraday momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate 3-day rolling standard deviation of intraday momentum
    data['momentum_std'] = data['intraday_momentum'].rolling(window=3, min_periods=2).std()
    
    # Derive reversal signal
    epsilon = 1e-8
    data['reversal_signal'] = -1 * data['intraday_momentum'] / (data['momentum_std'] + epsilon)
    
    # Calculate volume acceleration
    data['volume_median'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['volume_mad'] = (data['volume'] - data['volume_median']).abs().rolling(window=5, min_periods=3).median()
    data['volume_acceleration'] = (data['volume'] - data['volume_median']) / (data['volume_mad'] + 1e-8)
    
    # Calculate volume trend using rolling linear regression
    def rolling_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=3).apply(rolling_slope, raw=False)
    
    # Combine acceleration and trend using geometric mean
    data['volume_combination'] = np.sign(data['volume_acceleration']) * np.sqrt(np.abs(data['volume_acceleration'] * data['volume_trend']))
    
    # Generate composite alpha factor
    data['composite_factor'] = data['reversal_signal'] * data['volume_combination']
    
    # Apply conditional scaling based on absolute price change
    data['abs_price_change'] = (data['close'] - data['open']).abs()
    price_change_quantile = data['abs_price_change'].rolling(window=20, min_periods=10).quantile(0.5)
    data['scaling_factor'] = np.where(data['abs_price_change'] > price_change_quantile, 1.5, 1.0)
    data['scaled_factor'] = data['composite_factor'] * data['scaling_factor']
    
    # Multiply by amount to incorporate dollar volume information
    data['final_factor'] = data['scaled_factor'] * data['amount']
    
    return data['final_factor']
