import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Compute components for each day
    for i in range(len(data)):
        if i < 5:  # Need at least 5 days of history
            factor.iloc[i] = np.nan
            continue
            
        current = data.iloc[i]
        past_data = data.iloc[:i]  # Only use past data
        
        # 1. Intraday Price Momentum
        price_range_ratio = (current['high'] - current['low']) / current['close']
        
        # 2. High-Low Range Momentum
        # Calculate 5-day average high-low range
        last_5_days = past_data.iloc[-5:]
        avg_high_low_range = (last_5_days['high'] - last_5_days['low']).mean()
        
        # Calculate 20-day volatility (std of returns)
        if i >= 20:
            returns = past_data['close'].pct_change().dropna()
            last_20_returns = returns.iloc[-20:]
            volatility = last_20_returns.std()
        else:
            volatility = np.nan
            
        # Combine signals
        if not np.isnan(volatility) and volatility != 0:
            range_momentum = avg_high_low_range / volatility
        else:
            range_momentum = 0
            
        # Combine both signals
        combined_signal = price_range_ratio * range_momentum
        
        # Smooth with 5-day moving average
        if i >= 9:  # Need 5 more days for smoothing (current + 4 past)
            last_5_signals = factor.iloc[i-4:i+1]
            smoothed_signal = last_5_signals.mean()
        else:
            smoothed_signal = combined_signal
            
        factor.iloc[i] = smoothed_signal
    
    return factor
