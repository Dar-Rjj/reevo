import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining multiple market microstructure signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        # Current day data
        open_t = data.loc[current_date, 'open']
        high_t = data.loc[current_date, 'high']
        low_t = data.loc[current_date, 'low']
        close_t = data.loc[current_date, 'close']
        volume_t = data.loc[current_date, 'volume']
        
        # Previous day data
        close_t1 = data.loc[prev_date, 'close']
        high_t1 = data.loc[prev_date, 'high']
        low_t1 = data.loc[prev_date, 'low']
        
        # 1. Gap Reversal Momentum
        gap_size = (open_t - close_t1) / close_t1 if close_t1 != 0 else 0
        intraday_reversal = (close_t - open_t) / open_t if open_t != 0 else 0
        
        if abs(gap_size) > 1e-6:  # Avoid division by zero
            gap_reversal_ratio = intraday_reversal / gap_size
        else:
            gap_reversal_ratio = 0
            
        gap_reversal_signal = -gap_reversal_ratio  # Negative for reversal
        
        # 2. Volume-Price Divergence
        price_movement = (high_t - low_t) / open_t if open_t != 0 else 0
        if (high_t - low_t) > 1e-6:  # Avoid division by zero
            volume_concentration = volume_t / (high_t - low_t)
        else:
            volume_concentration = 0
            
        divergence_score = price_movement * volume_concentration
        divergence_signal = -divergence_score  # Negative for divergence
        
        # 3. Opening Momentum Persistence
        opening_momentum = (open_t - close_t1) / close_t1 if close_t1 != 0 else 0
        closing_momentum = (close_t - open_t) / open_t if open_t != 0 else 0
        
        if abs(opening_momentum) > 1e-6:  # Avoid division by zero
            momentum_persistence = closing_momentum / opening_momentum
        else:
            momentum_persistence = 0
            
        persistence_signal = momentum_persistence
        
        # 4. Range Expansion Efficiency
        prev_range = (high_t1 - low_t1) / close_t1 if close_t1 != 0 else 0
        current_range = (high_t - low_t) / open_t if open_t != 0 else 0
        
        if prev_range > 1e-6:  # Avoid division by zero
            range_expansion = current_range / prev_range
        else:
            range_expansion = 0
            
        expansion_signal = -range_expansion  # Negative for mean reversion
        
        # 5. Intraday Pressure Reversal
        morning_pressure = (high_t - open_t) / open_t if open_t != 0 else 0
        afternoon_pressure = (close_t - low_t) / close_t if close_t != 0 else 0
        pressure_reversal = afternoon_pressure - morning_pressure
        pressure_signal = pressure_reversal
        
        # Combine all signals with equal weights
        combined_factor = (
            gap_reversal_signal + 
            divergence_signal + 
            persistence_signal + 
            expansion_signal + 
            pressure_signal
        )
        
        factor_values.loc[current_date] = combined_factor
    
    # Fill first day with NaN since we need previous day data
    if len(factor_values) > 0:
        factor_values.iloc[0] = np.nan
    
    return factor_values
