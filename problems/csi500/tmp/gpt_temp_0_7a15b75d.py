import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for momentum analysis
    data['returns_5d'] = data['close'].pct_change(5)
    data['returns_20d'] = data['close'].pct_change(20)
    
    # Momentum ratio (short/long)
    data['momentum_ratio'] = (1 + data['returns_5d']) / (1 + data['returns_20d']) - 1
    
    # Calculate daily range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # 5-day average range for volatility compression analysis
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    
    # Range compression ratio
    data['range_compression'] = data['daily_range'] / data['avg_range_5d'].shift(1)
    
    # Historical percentile of current range (20-day lookback)
    data['range_percentile'] = data['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Consecutive low volatility days (range < 50th percentile)
    low_vol_mask = data['daily_range'] < data['daily_range'].rolling(window=20).quantile(0.5)
    data['consecutive_low_vol'] = low_vol_mask.groupby((~low_vol_mask).cumsum()).cumcount() + 1
    
    # Compression-to-expansion ratio
    data['compression_expansion'] = data['daily_range'] / data['daily_range'].shift(1)
    
    # Volume analysis during compression periods
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5d'].shift(1)
    
    # Momentum divergence patterns
    data['price_position'] = (data['close'] - data['low'].rolling(window=20).min()) / \
                            (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min())
    
    # Positive divergence: low price position but high momentum
    data['positive_divergence'] = ((data['price_position'] < 0.3) & 
                                  (data['momentum_ratio'] > data['momentum_ratio'].rolling(window=20).quantile(0.7)))
    
    # Negative divergence: high price position but low momentum
    data['negative_divergence'] = ((data['price_position'] > 0.7) & 
                                  (data['momentum_ratio'] < data['momentum_ratio'].rolling(window=20).quantile(0.3)))
    
    # Divergence magnitude
    data['momentum_spread'] = data['returns_5d'] - data['returns_20d']
    
    # Volatility compression ratio (inverse of range compression)
    data['vol_compression_ratio'] = 1 / data['range_compression']
    
    # Time since last volatility expansion
    vol_expansion = data['daily_range'] > data['daily_range'].rolling(window=10).quantile(0.7)
    data['days_since_expansion'] = vol_expansion.groupby(vol_expansion.cumsum()).cumcount()
    
    # Duration of momentum divergence
    positive_div_mask = data['positive_divergence'] | data['negative_divergence']
    data['divergence_duration'] = positive_div_mask.groupby((~positive_div_mask).cumsum()).cumcount()
    
    # Construct final alpha factor
    # Base signal: positive for positive divergence with compression, negative for negative divergence with expansion
    base_signal = np.zeros(len(data))
    
    # Positive signals
    pos_condition = (data['positive_divergence'] & 
                    (data['range_compression'] < 0.8) & 
                    (data['consecutive_low_vol'] >= 3))
    
    # Negative signals  
    neg_condition = (data['negative_divergence'] & 
                    (data['range_compression'] > 1.2) & 
                    (data['volume_ratio'] > 1.5))
    
    base_signal[pos_condition] = 1
    base_signal[neg_condition] = -1
    
    # Weight by divergence strength and compression duration
    divergence_strength = data['momentum_spread'].abs()
    compression_strength = (1 / data['range_compression']).clip(upper=5)
    duration_weight = np.log1p(data['divergence_duration'])
    
    # Final factor calculation
    factor = base_signal * divergence_strength * compression_strength * duration_weight
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
