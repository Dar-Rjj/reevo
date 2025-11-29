import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday momentum decay with volume acceleration
    # Morning momentum (Open to Midday) - assuming midday is 12:00, using (high+low)/2 as proxy
    data['midday_price'] = (data['high'] + data['low']) / 2
    data['morning_momentum'] = (data['midday_price'] - data['open']) / data['open']
    
    # Afternoon momentum (Midday to Close)
    data['afternoon_momentum'] = (data['close'] - data['midday_price']) / data['midday_price']
    
    # Momentum decay ratio
    data['momentum_decay_ratio'] = data['afternoon_momentum'] / (data['morning_momentum'] + 1e-8)
    
    # 3-day rolling momentum persistence
    data['momentum_persistence_3d'] = data['morning_momentum'].rolling(window=3).mean()
    
    # 10-day rolling momentum acceleration
    data['momentum_acceleration_10d'] = data['morning_momentum'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 10 else np.nan, raw=False
    )
    
    # Momentum decay signal (ratio of short-term to long-term momentum decay)
    short_term_decay = data['momentum_decay_ratio'].rolling(window=3).mean()
    long_term_decay = data['momentum_decay_ratio'].rolling(window=10).mean()
    data['momentum_decay_signal'] = short_term_decay / (long_term_decay + 1e-8)
    
    # Volume acceleration
    # Assuming we can calculate intraday volume patterns using daily volume as proxy
    data['morning_volume_accel'] = data['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8) if len(x) == 5 else np.nan, raw=False
    )
    data['afternoon_volume_accel'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8) if len(x) == 3 else np.nan, raw=False
    )
    data['volume_accel_ratio'] = data['afternoon_volume_accel'] / (data['morning_volume_accel'] + 1e-8)
    
    # Combine momentum decay with volume acceleration
    data['intraday_momentum_factor'] = data['momentum_decay_signal'] * data['volume_accel_ratio']
    
    # Price range expansion with volatility compression
    # 5-day intraday range expansion
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_expansion_5d'] = data['daily_range'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) == 5 else np.nan, raw=False
    )
    
    # 20-day intraday range compression
    data['range_compression_20d'] = data['daily_range'].rolling(window=20).apply(
        lambda x: (x.mean() - x.iloc[-1]) / (x.std() + 1e-8) if len(x) == 20 else np.nan, raw=False
    )
    
    # Volatility clustering and mean reversion
    returns = data['close'].pct_change()
    data['volatility_clustering'] = returns.rolling(window=10).std()
    data['volatility_mean_reversion'] = data['volatility_clustering'].rolling(window=20).apply(
        lambda x: (x.mean() - x.iloc[-1]) / (x.std() + 1e-8) if len(x) == 20 else np.nan, raw=False
    )
    
    # Range-volatility signal
    data['range_volatility_signal'] = data['range_expansion_5d'] * data['range_compression_20d']
    
    # Price level anchoring
    data['recent_high'] = data['high'].rolling(window=10).max()
    data['recent_low'] = data['low'].rolling(window=10).min()
    data['price_anchoring'] = 1 - abs((data['close'] - (data['recent_high'] + data['recent_low']) / 2) / 
                                    ((data['recent_high'] - data['recent_low']) + 1e-8))
    
    data['price_range_factor'] = data['range_volatility_signal'] * data['price_anchoring']
    
    # Opening momentum persistence with volume confirmation
    # Gap momentum
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # First-hour momentum persistence (using morning momentum as proxy)
    data['first_hour_persistence'] = data['morning_momentum'].rolling(window=3).mean()
    
    # Midday momentum continuation
    data['midday_continuation'] = data['afternoon_momentum'].rolling(window=3).mean()
    
    # Closing momentum convergence
    data['closing_convergence'] = (data['close'] - data['open']) / data['open']
    
    # Momentum persistence signal
    data['momentum_persistence_signal'] = data['gap_momentum'] * data['first_hour_persistence']
    
    # Volume confirmation
    data['volume_momentum_alignment'] = data['volume'].rolling(window=5).corr(data['close'].pct_change())
    
    data['opening_momentum_factor'] = data['momentum_persistence_signal'] * data['volume_momentum_alignment']
    
    # Combine all factors with equal weighting
    data['final_factor'] = (
        data['intraday_momentum_factor'].fillna(0) +
        data['price_range_factor'].fillna(0) +
        data['opening_momentum_factor'].fillna(0)
    ) / 3
    
    return data['final_factor']
