import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lagged values
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Remove first row with NaN values from lagged columns
    data = data.iloc[1:]
    
    for date in data.index:
        row = data.loc[date]
        
        # Gap Reversal Momentum
        gap_size = (row['open'] - row['prev_close']) / row['prev_close']
        intraday_reversal = (row['close'] - row['open']) / row['open']
        if gap_size != 0:
            gap_reversal_ratio = intraday_reversal / gap_size
        else:
            gap_reversal_ratio = 0
        
        # Volume-Price Divergence
        price_movement = (row['high'] - row['low']) / row['open']
        if (row['high'] - row['low']) != 0:
            volume_concentration = row['volume'] / (row['high'] - row['low'])
        else:
            volume_concentration = 0
        divergence_score = price_movement * volume_concentration
        
        # Opening Momentum Persistence
        opening_momentum = (row['open'] - row['prev_close']) / row['prev_close']
        closing_momentum = (row['close'] - row['open']) / row['open']
        if opening_momentum != 0:
            momentum_persistence = closing_momentum / opening_momentum
        else:
            momentum_persistence = 0
        
        # Range Expansion Efficiency
        previous_range = (row['prev_high'] - row['prev_low']) / row['prev_close']
        current_range = (row['high'] - row['low']) / row['open']
        if previous_range != 0:
            range_expansion = current_range / previous_range
        else:
            range_expansion = 0
        
        # Intraday Pressure Reversal
        morning_pressure = (row['high'] - row['open']) / row['open']
        afternoon_pressure = (row['close'] - row['low']) / row['close']
        pressure_reversal = afternoon_pressure - morning_pressure
        
        # Combine factors (equal weighting for simplicity)
        combined_factor = (
            gap_reversal_ratio + 
            divergence_score + 
            momentum_persistence + 
            range_expansion + 
            pressure_reversal
        )
        
        result[date] = combined_factor
    
    return result
