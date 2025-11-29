import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily high-low range
    data['daily_range'] = data['high'] - data['low']
    
    # 1. Identify Volatility Regime Shifts
    # Calculate 5-day rolling realized volatility using high-low range
    data['vol_5d'] = data['daily_range'].rolling(window=5).std()
    
    # Compute 20-day rolling median volatility
    data['vol_20d_median'] = data['daily_range'].rolling(window=20).median()
    
    # Compute volatility regime ratio
    data['vol_regime_ratio'] = data['vol_5d'] / data['vol_20d_median']
    
    # Derive regime signal using sigmoid transformation
    data['regime_signal'] = 1 / (1 + np.exp(-data['vol_regime_ratio']))
    
    # 2. Detect Price-Volume Divergence
    # Calculate price momentum as (Close - Open) / (High - Low)
    data['price_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Compute volume momentum as z-score of volume
    data['volume_5d_mean'] = data['volume'].rolling(window=5).mean()
    data['volume_5d_std'] = data['volume'].rolling(window=5).std()
    data['volume_momentum'] = (data['volume'] - data['volume_5d_mean']) / data['volume_5d_std']
    
    # Calculate correlation between price momentum and volume momentum over 10-day window
    data['price_volume_corr'] = data['price_momentum'].rolling(window=10).corr(data['volume_momentum'])
    
    # Derive divergence signal
    data['divergence_signal'] = -1 * data['price_volume_corr'] * data['price_momentum']
    
    # 3. Construct Adaptive Alpha Factor
    # Multiply regime signal by divergence signal
    data['raw_factor'] = data['regime_signal'] * data['divergence_signal']
    
    # Apply volatility-adjusted scaling using 20-day rolling volatility
    data['vol_20d'] = data['daily_range'].rolling(window=20).std()
    data['vol_adjusted_factor'] = data['raw_factor'] / data['vol_20d']
    
    # Incorporate amount-based weighting using rank normalization
    data['amount_rank'] = data['amount'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Final factor with amount weighting
    data['factor'] = data['vol_adjusted_factor'] * data['amount_rank']
    
    # Return the factor series
    return data['factor']
