import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Intraday Price Reversal Signal
    # Normalized Price Range
    norm_price_range = (data['high'] - data['low']) / data['close']
    
    # Volume-Scaled Reversal
    avg_volume_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_scaled_reversal = norm_price_range * data['volume'] / avg_volume_5d
    
    # 2. Volatility-Adjusted Trend
    # Short-Term Momentum
    daily_returns = data['close'].pct_change()
    momentum_1d = data['close'] / data['close'].shift(1)
    
    # Volatility Scaling
    vol_5d = daily_returns.rolling(window=5, min_periods=1).std()
    vol_adjusted_trend = momentum_1d / vol_5d.replace(0, np.nan)
    
    # 3. Combine Components
    raw_factor = volume_scaled_reversal * vol_adjusted_trend
    
    # Liquidity Adjustment
    log_volume = np.log(data['volume'].replace(0, np.nan))
    
    # Market Volume Percentile (20D)
    def rolling_percentile(s):
        return s.rolling(window=20, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    volume_percentile = rolling_percentile(data['volume'])
    
    # Final Factor Calculation
    factor = raw_factor * log_volume / volume_percentile.replace(0, np.nan)
    
    return factor
