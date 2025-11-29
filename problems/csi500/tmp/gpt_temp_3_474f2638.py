import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Intraday Momentum Components
    # Short-term intraday trend: (Close - Open) / Open
    intraday_trend = (data['close'] - data['open']) / data['open']
    
    # Medium-term momentum: 5-day rolling close price change
    medium_term_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Momentum divergence: short-term vs medium-term comparison
    momentum_divergence = intraday_trend - medium_term_momentum
    
    # Volume Confirmation System
    # Abnormal volume detection: Volume / 5-day rolling volume average
    volume_5day_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    abnormal_volume = data['volume'] / volume_5day_avg
    
    # Volume-price relationship: 5-day rolling correlation between volume and price changes
    price_change = data['close'].pct_change()
    volume_price_corr = data['volume'].rolling(window=5, min_periods=1).corr(price_change)
    
    # Reversal Signal Generation
    # Extreme intraday movers: High-to-Close Return = (High - Close) / Close
    high_to_close_return = (data['high'] - data['close']) / data['close']
    
    # Volume-confirmed reversal: combine extreme moves with volume spikes
    extreme_moves = high_to_close_return.rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    volume_confirmed_reversal = extreme_moves * abnormal_volume
    
    # Factor Integration
    # Combine momentum divergence with reversal signals
    momentum_reversal_combined = momentum_divergence * volume_confirmed_reversal
    
    # Apply volume confirmation as weighting mechanism
    volume_weight = np.tanh(abnormal_volume - 1)  # Scale abnormal volume to [-1, 1] range
    final_factor = momentum_reversal_combined * volume_weight
    
    # Apply volume-price correlation as additional confirmation
    final_factor = final_factor * np.tanh(volume_price_corr)
    
    return final_factor
