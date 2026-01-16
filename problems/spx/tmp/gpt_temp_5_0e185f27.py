import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Compute components
    for t in range(len(df)):
        current_data = df.iloc[:t+1]  # Only use current and past data
        
        # 1. Price Range Momentum component
        price_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        
        # Volume Slope (5-day)
        if t >= 4:
            volume_slope = np.polyfit(range(5), current_data['volume'].iloc[-5:], 1)[0]
        else:
            volume_slope = 0  # Not enough data
            
        momentum_component = price_range * volume_slope
        
        # 2. Price Reversal component
        if t >= 3:
            price_reversal = current_data['close'].iloc[-1] - current_data['close'].iloc[-4]
            
            # Volume Ratio (current volume / 5-day MA)
            if t >= 4:
                vol_ma = current_data['volume'].iloc[-5:].mean()
            else:
                vol_ma = current_data['volume'].iloc[:t+1].mean()
            
            volume_ratio = current_data['volume'].iloc[-1] / vol_ma if vol_ma != 0 else 0
            reversal_component = price_reversal * volume_ratio
        else:
            reversal_component = 0
            
        # 3. Volatility normalization
        if t >= 4:
            price_std = current_data['close'].iloc[-5:].std()
        else:
            price_std = 1  # Avoid division by zero
            
        # Combine components and normalize
        if price_std != 0:
            combined = (momentum_component + reversal_component) / price_std
        else:
            combined = 0
            
        factor.iloc[t] = combined
    
    return factor
