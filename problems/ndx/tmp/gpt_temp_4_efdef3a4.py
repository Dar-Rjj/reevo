import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        # Get current day data
        current_open = data.loc[current_date, 'open']
        current_high = data.loc[current_date, 'high']
        current_low = data.loc[current_date, 'low']
        current_close = data.loc[current_date, 'close']
        current_volume = data.loc[current_date, 'volume']
        
        # Get previous day close
        prev_close = data.loc[prev_date, 'close']
        
        # Calculate first hour return (assuming first hour high is current_high)
        first_hour_return = (current_high - current_open) / current_open
        
        # Calculate 5-day average volume (using only past data)
        if i >= 5:
            past_5_days = data.index[i-5:i]
            avg_volume_5day = data.loc[past_5_days, 'volume'].mean()
            volume_confirmation = current_volume / avg_volume_5day
        else:
            volume_confirmation = 1.0  # neutral value when not enough history
        
        # Morning momentum score
        morning_momentum = first_hour_return * volume_confirmation
        
        # Midday pullback (using current low as midday low)
        midday_pullback = (current_low - prev_close) / prev_close
        
        # Closing strength
        closing_strength = (current_close - current_low) / current_low if current_low != 0 else 0
        
        # Afternoon reversal signal
        afternoon_reversal = midday_pullback * closing_strength
        
        # Combined factor
        factor.loc[current_date] = morning_momentum * afternoon_reversal
    
    return factor
