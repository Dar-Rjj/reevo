import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate components for each day
    for i in range(2, len(data)):
        current = data.iloc[i]
        prev_close = data.iloc[i-1]['close']
        prev_prev_close = data.iloc[i-2]['close']
        
        # Momentum Reversal Components
        opening_gap = (current['open'] - prev_prev_close) / prev_prev_close
        closing_reversal = (current['close'] - current['open']) / current['open']
        momentum_reversal = opening_gap + closing_reversal
        
        # Volume Adjustment
        if i >= 10:  # Need at least 10 days history for volume median
            volume_history = data.iloc[max(0, i-10):i]['volume']
            median_volume = volume_history.median()
            volume_spike = current['volume'] / median_volume
            log_volume = np.log(volume_spike + 1)
        else:
            log_volume = 1  # Neutral value when insufficient history
            
        # Volatility Normalization
        if i >= 10:  # Need at least 10 days for rolling std
            returns = data.iloc[max(0, i-10):i]['close'].pct_change().dropna()
            vol = returns.std()
            if vol > 0:
                momentum_reversal /= vol
                
        # Combine signals
        factor_value = momentum_reversal * log_volume
        
        # Winsorize extreme values (top and bottom 1%)
        if i >= 20:  # Need some history to determine percentiles
            recent_values = factor.iloc[max(0, i-20):i].dropna()
            if len(recent_values) > 0:
                lower = recent_values.quantile(0.01)
                upper = recent_values.quantile(0.99)
                factor_value = np.clip(factor_value, lower, upper)
        
        factor.iloc[i] = factor_value
    
    return factor
