import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Normalized Intraday Return
    denominator = data['open'] + data['close']
    # Avoid division by zero by replacing zeros with NaN
    normalized_intraday_return = (data['high'] - data['low']) / np.where(denominator != 0, denominator, np.nan)
    
    # Compute Volume-Adjusted Price Change
    volume_adjusted_price_change = (data['close'] - data['open']) * data['volume']
    
    # Calculate rolling 5-day standard deviation of Volume-Adjusted Price Change
    vol_adj_price_std = volume_adjusted_price_change.rolling(window=5, min_periods=3).std()
    
    # Calculate 3-day lagged correlation between signals
    # Use rolling correlation with window that includes lag
    correlation_window = 5  # Window size for correlation calculation
    
    def calculate_lagged_correlation(norm_return, vol_adj_change, window):
        correlations = []
        for i in range(len(norm_return)):
            if i < window:
                correlations.append(np.nan)
            else:
                # Use current and past data only
                window_norm = norm_return.iloc[i-window+1:i+1]
                window_vol = vol_adj_change.iloc[i-window+1-3:i+1-3]  # 3-day lag
                # Ensure both windows have same length after lag
                min_len = min(len(window_norm), len(window_vol))
                if min_len >= 3:  # Minimum periods for meaningful correlation
                    window_norm = window_norm.iloc[-min_len:]
                    window_vol = window_vol.iloc[-min_len:]
                    corr = window_norm.corr(window_vol)
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
        return pd.Series(correlations, index=norm_return.index)
    
    lagged_correlation = calculate_lagged_correlation(
        normalized_intraday_return, 
        volume_adjusted_price_change, 
        correlation_window
    )
    
    # Calculate Volume-to-Amount Ratio with zero handling
    volume_to_amount = data['volume'] / np.where(data['amount'] != 0, data['amount'], np.nan)
    
    # Combine signals
    factor = lagged_correlation * volume_to_amount
    
    return factor
