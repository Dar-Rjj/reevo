import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate components
    for t in range(len(data)):
        current_data = data.iloc[:t+1]  # Only use current and past data
        
        if t == 0:
            continue  # Skip first day as we need previous data for calculations
        
        # 1. Directional Intraday Momentum
        open_price = current_data.iloc[-1]['open']
        close_price = current_data.iloc[-1]['close']
        high_price = current_data.iloc[-1]['high']
        low_price = current_data.iloc[-1]['low']
        
        # Avoid division by zero
        if high_price != low_price:
            raw_momentum = (close_price - open_price) / (high_price - low_price)
            direction = np.sign(close_price - open_price)
            normalized_momentum = raw_momentum * direction
        else:
            normalized_momentum = 0.0
        
        # 2. Volatility Adjustment
        if t >= 20:  # Need at least 20 days of history
            # Calculate daily returns
            daily_returns = current_data['close'].pct_change().dropna()
            # Use past 20 days (excluding current day)
            vol_window = daily_returns.iloc[-20:-1] if t > 20 else daily_returns.iloc[:-1]
            volatility = vol_window.std()
            # Adjust momentum with volatility (add small epsilon to avoid division by zero)
            adj_momentum = normalized_momentum / (volatility + 1e-6)
        else:
            adj_momentum = normalized_momentum  # No adjustment if not enough history
        
        # 3. Volume Confirmation
        current_volume = current_data.iloc[-1]['volume']
        if t >= 20:
            # Use past 20 days volume (excluding current day)
            vol_window = current_data['volume'].iloc[-21:-1] if t > 20 else current_data['volume'].iloc[:-1]
            avg_volume = vol_window.mean()
            volume_ratio = current_volume / (avg_volume + 1e-6)
        else:
            volume_ratio = 1.0  # Neutral if not enough history
        
        # Combine components
        factor.iloc[t] = adj_momentum * volume_ratio
    
    return factor
