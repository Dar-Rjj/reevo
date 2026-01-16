import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each day using only past data
    for t in range(1, len(df)):
        current_data = df.iloc[:t+1]  # All data up to and including current day
        
        # 1. Measure Intraday Momentum
        # Price Range Normalization
        high = current_data['high'].iloc[-1]
        low = current_data['low'].iloc[-1]
        close = current_data['close'].iloc[-1]
        open_ = current_data['open'].iloc[-1]
        
        price_range = (high - low) / (high + low) * (close / open_)
        
        # Volume Adjustment
        volume = current_data['volume'].iloc[-1]
        if t >= 5:
            mean_volume = current_data['volume'].iloc[-5:].mean()
        else:
            mean_volume = np.nan
            
        if not np.isnan(mean_volume) and mean_volume > 0:
            volume_adj = np.sqrt(volume) / mean_volume
        else:
            volume_adj = 0
            
        momentum = price_range * volume_adj
        
        # 2. Incorporate Mean-Reversion
        # Price Deviation
        if t >= 10:
            sma_10 = current_data['close'].iloc[-10:].mean()
            price_dev = (close - sma_10) / close
            
            # Normalize by 10-day Price Range Mean
            price_ranges = (current_data['high'].iloc[-10:] - current_data['low'].iloc[-10:]) / \
                          (current_data['high'].iloc[-10:] + current_data['low'].iloc[-10:])
            range_mean = price_ranges.mean()
            
            if range_mean > 0:
                price_dev_normalized = price_dev / range_mean
            else:
                price_dev_normalized = 0
                
            # Volatility Adjustment
            returns = np.log(current_data['close'].iloc[-10:]) - np.log(current_data['close'].iloc[-11:-1])
            vol_10 = returns.std()
            
            if vol_10 > 0:
                mean_reversion = price_dev_normalized / vol_10
            else:
                mean_reversion = 0
        else:
            mean_reversion = 0
            
        # 3. Combine Components
        combined = momentum * mean_reversion
        
        # Normalize by 10-day Std Dev if available
        if t >= 10:
            combined_values = factor.iloc[-10:-1].dropna()
            if len(combined_values) > 0:
                combined_std = combined_values.std()
                if combined_std > 0:
                    combined_normalized = combined / combined_std
                else:
                    combined_normalized = 0
            else:
                combined_normalized = 0
        else:
            combined_normalized = 0
            
        factor.iloc[t] = combined_normalized
    
    return factor
