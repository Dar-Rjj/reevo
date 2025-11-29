import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price components
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    
    # Breakout Component
    rolling_max_high = high.rolling(window=5, min_periods=1).max()
    rolling_min_low = low.rolling(window=5, min_periods=1).min()
    breakout_component = (high / rolling_max_high) - (low / rolling_min_low)
    
    # Reversal Component
    price_range = high - low
    # Avoid division by zero
    price_range = price_range.replace(0, np.nan)
    
    normalized_close_open = (close - open_price) / price_range
    normalized_close_mid = (close - (high + low) / 2) / price_range
    
    # Calculate 3-day rolling correlation
    correlation_window = 3
    reversal_corr = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= correlation_window - 1:
            start_idx = i - correlation_window + 1
            end_idx = i + 1
            
            window_nco = normalized_close_open.iloc[start_idx:end_idx]
            window_ncm = normalized_close_mid.iloc[start_idx:end_idx]
            
            # Calculate correlation only if we have enough non-NaN values
            valid_mask = (~window_nco.isna()) & (~window_ncm.isna())
            if valid_mask.sum() >= 2:
                reversal_corr.iloc[i] = window_nco[valid_mask].corr(window_ncm[valid_mask])
            else:
                reversal_corr.iloc[i] = 0
        else:
            reversal_corr.iloc[i] = 0
    
    diff_abs = abs(normalized_close_open - normalized_close_mid)
    reversal_component = reversal_corr * diff_abs
    
    # Breakout-Reversal Signal
    breakout_reversal = breakout_component * reversal_component
    
    # Volume Confirmation
    # Volume Velocity
    price_change_direction = np.sign((close - open_price) * volume)
    volume_velocity = price_change_direction * (volume / volume.rolling(window=3, min_periods=1).mean())
    
    # Volume Momentum
    numerator = ((close - open_price) * volume).rolling(window=5, min_periods=1).sum()
    denominator = volume.rolling(window=5, min_periods=1).sum()
    volume_momentum = numerator / denominator.replace(0, np.nan)
    
    volume_confirmation = volume_velocity * volume_momentum
    
    # Final Alpha Factor
    alpha_factor = breakout_reversal * volume_confirmation
    
    return alpha_factor
