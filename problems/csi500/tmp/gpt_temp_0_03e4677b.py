import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday range efficiency
    # For simplicity, we'll approximate morning and afternoon ranges using available data
    # Morning range utilization: (High - Open) / Open
    morning_utilization = (data['high'] - data['open']) / data['open']
    
    # Afternoon range performance: (Close - Low) / Close
    afternoon_performance = (data['close'] - data['low']) / data['close']
    
    # Combine intraday range components
    intraday_range_efficiency = morning_utilization * afternoon_performance
    
    # Calculate volatility-adjusted momentum reversal
    # Range momentum divergence
    price_momentum = data['close'].pct_change(periods=1)
    range_momentum = ((data['high'] - data['low']) / data['close']).pct_change(periods=1)
    divergence_signal = np.sign(range_momentum) * np.sign(price_momentum)
    
    # Volatility adjustment
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    rolling_true_range = true_range.rolling(window=10, min_periods=5).mean()
    
    # Price reversal from 3-day high/low
    rolling_high_3d = data['high'].rolling(window=3, min_periods=2).max()
    rolling_low_3d = data['low'].rolling(window=3, min_periods=2).min()
    price_reversal = (data['close'] - rolling_low_3d) / (rolling_high_3d - rolling_low_3d + 1e-8)
    
    # Volatility-adjusted reversal
    volatility_adjusted_reversal = price_reversal / (rolling_true_range + 1e-8)
    
    # Combine momentum with volatility adjustment
    momentum_reversal = divergence_signal * volatility_adjusted_reversal
    
    # Apply persistence weighting (using 5-day momentum persistence)
    momentum_persistence = momentum_reversal.rolling(window=5, min_periods=3).mean()
    volatility_adjusted_momentum = momentum_reversal * momentum_persistence
    
    # Evaluate volume-weighted range volatility context
    # Short-term range volatility
    daily_range = data['high'] - data['low']
    short_term_volatility = daily_range.rolling(window=5, min_periods=3).std()
    
    # Volume confirmation
    volume_mean_5d = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_ratio = data['volume'] / (volume_mean_5d + 1e-8)
    volume_confirmation = np.log(volume_ratio + 1)
    
    # Volume-weighted volatility ratio
    volume_weighted_volatility = short_term_volatility * volume_confirmation
    long_term_volatility = daily_range.rolling(window=20, min_periods=10).std()
    volatility_ratio = volume_weighted_volatility / (long_term_volatility + 1e-8)
    
    # Generate final alpha factor
    alpha_factor = (intraday_range_efficiency * 
                   volatility_adjusted_momentum * 
                   volatility_ratio)
    
    # Apply contrarian effect for extreme range behavior
    range_extremes = (daily_range / data['close']).rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    contrarian_multiplier = 1 - np.tanh(np.abs(range_extremes))
    
    final_alpha = alpha_factor * contrarian_multiplier
    
    return final_alpha
