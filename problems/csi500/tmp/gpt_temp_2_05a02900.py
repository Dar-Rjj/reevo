import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Breakout Momentum Efficiency
    # Breakout Direction
    high_breakout = (data['close'] - data['high']) / data['high']
    low_breakout = (data['close'] - data['low']) / data['low']
    breakout_direction = np.where(data['close'] > (data['high'] + data['low']) / 2, 
                                 high_breakout, low_breakout)
    
    # Breakout Momentum
    intraday_momentum = (data['close'] - data['open']) / data['open']
    price_range = (data['high'] - data['low']) / data['open']
    
    # Efficiency Ratio
    momentum_efficiency = intraday_momentum / np.where(price_range != 0, price_range, 1)
    volume_intensity = data['volume'] / data['volume'].rolling(window=20, min_periods=5).mean()
    breakout_efficiency = momentum_efficiency * volume_intensity
    
    factor1 = breakout_direction * breakout_efficiency
    
    # Factor 2: Volume-Confirmed Acceleration
    # Price Acceleration
    daily_return = data['close'].pct_change()
    return_acceleration = daily_return - daily_return.rolling(window=5, min_periods=3).mean()
    
    # Volume Momentum
    volume_change = data['volume'].pct_change()
    relative_volume = data['volume'] / data['volume'].rolling(window=20, min_periods=5).mean()
    
    # Signal Combination
    acceleration_volume = return_acceleration * relative_volume
    volatility = data['close'].rolling(window=20, min_periods=5).std()
    volatility_scaled = acceleration_volume / np.where(volatility != 0, volatility, 1)
    
    factor2 = volatility_scaled
    
    # Factor 3: Gap Persistence with Volatility
    # Gap Characteristics
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    daily_range = (data['high'] - data['low']) / data['open']
    
    # Persistence Efficiency
    intraday_persistence = (data['close'] - data['open']) / np.where(gap != 0, gap, 1)
    gap_efficiency = gap / np.where(daily_range != 0, daily_range, 1)
    volume_confirmation = data['volume'] / data['volume'].rolling(window=10, min_periods=3).mean()
    
    factor3 = gap_efficiency * intraday_persistence * volume_confirmation
    
    # Factor 4: Trend Strength with Breakout
    # Short-term Trend
    price_slope = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan
    )
    
    # Breakout Confirmation
    intraday_breakout = (data['close'] - (data['high'] + data['low']) / 2) / data['open']
    breakout_efficiency_2 = intraday_momentum / np.where(price_range != 0, price_range, 1)
    
    # Signal Combination
    trend_breakout = price_slope * intraday_breakout * breakout_efficiency_2
    volatility_2 = data['close'].rolling(window=10, min_periods=3).std()
    volume_weight = data['volume'] / data['volume'].rolling(window=10, min_periods=3).mean()
    
    factor4 = trend_breakout * volume_weight / np.where(volatility_2 != 0, volatility_2, 1)
    
    # Combine all factors with equal weighting
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + 
                      factor3.fillna(0) + factor4.fillna(0)) / 4
    
    return combined_factor
