import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate components for each day
    for t in range(len(data)):
        current_data = data.iloc[:t+1].copy()  # Only use data up to current day
        
        # 1. Intraday Momentum Component
        # Normalized Price Move
        price_move = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'] + 1e-6)
        price_move = np.clip(price_move, -1, 1)
        
        # Adjust by Volatility (5-day StdDev of High-Low range)
        if t >= 4:
            high_low_range = current_data['high'] - current_data['low']
            volatility = high_low_range.rolling(5).std().iloc[t]
            adjusted_move = price_move.iloc[t] / (volatility + 1e-6)
        else:
            adjusted_move = 0
        
        # 2. Volume Confirmation
        # Volume Z-Score (20-day)
        if t >= 19:
            volume_mean = current_data['volume'].rolling(20).mean().iloc[t]
            volume_std = current_data['volume'].rolling(20).std().iloc[t]
            volume_z = (current_data['volume'].iloc[t] - volume_mean) / (volume_std + 1e-6)
            # Sigmoid function
            volume_confirmation = 1 / (1 + np.exp(-volume_z))
        else:
            volume_confirmation = 0
        
        # Combine components
        factor.iloc[t] = adjusted_move * volume_confirmation
    
    # Final Processing
    # Cross-sectional rank
    ranked_factor = factor.groupby(factor.index).rank(pct=True)
    
    # Standardize to Z-score (using expanding window)
    mean = ranked_factor.expanding().mean()
    std = ranked_factor.expanding().std()
    final_factor = (ranked_factor - mean) / (std + 1e-6)
    
    return final_factor
