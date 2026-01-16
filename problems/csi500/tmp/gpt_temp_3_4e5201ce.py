import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Price Momentum Component
    close = data['close']
    log_returns = np.log(close / close.shift(1))
    volatility = log_returns.rolling(window=5, min_periods=5).std()
    price_change = close - close.shift(5)
    normalized_momentum = price_change / (volatility * close.shift(5))
    
    # Volume Confirmation Component
    volume = data['volume']
    volume_trend = volume / volume.shift(5) - 1
    adjusted_momentum = normalized_momentum * volume_trend
    
    # Final Factor Construction
    historical_range = adjusted_momentum.rolling(window=20, min_periods=20).max() - adjusted_momentum.rolling(window=20, min_periods=20).min()
    normalized_factor = adjusted_momentum / historical_range
    
    # Check correlation with returns (using past data only)
    lookback_corr = 20
    corr = []
    for t in range(len(normalized_factor)):
        if t >= lookback_corr:
            window = normalized_factor.iloc[t-lookback_corr+1:t+1]
            ret_window = log_returns.iloc[t-lookback_corr+1:t+1]
            corr_val = window.corr(ret_window)
            corr.append(corr_val)
        else:
            corr.append(np.nan)
    
    # Create Series for correlation
    corr_series = pd.Series(corr, index=normalized_factor.index)
    
    # Scale factor based on correlation sign
    final_factor = normalized_factor * np.where(corr_series < 0, -1, 1)
    
    return final_factor
