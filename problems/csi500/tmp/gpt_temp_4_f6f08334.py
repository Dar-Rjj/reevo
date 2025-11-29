import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily high-low range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate range expansion/contraction
    df['range_change'] = df['daily_range'].pct_change()
    
    # Detect range expansion streaks
    expansion_mask = df['range_change'] > 0
    df['expansion_streak'] = 0
    
    # Calculate consecutive expansion days
    streak = 0
    for i in range(len(df)):
        if expansion_mask.iloc[i]:
            streak += 1
        else:
            streak = 0
        df.iloc[i, df.columns.get_loc('expansion_streak')] = streak
    
    # Range contraction signal: range decreases after expansion streak
    contraction_mask = (df['range_change'] < 0) & (df['expansion_streak'].shift(1) >= 1)
    
    # Volume confirmation: volume decreases during contraction
    df['volume_change'] = df['volume'].pct_change()
    volume_confirmation = (df['volume_change'] < 0) & contraction_mask
    
    # Calculate factor value
    # Higher value when strong expansion streak followed by contraction with volume confirmation
    factor = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if volume_confirmation.iloc[i]:
            # Factor increases with longer expansion streak and larger volume decrease
            expansion_streak_length = df['expansion_streak'].shift(1).iloc[i]
            volume_decrease_magnitude = abs(df['volume_change'].iloc[i])
            range_contraction_magnitude = abs(df['range_change'].iloc[i])
            
            factor[i] = (expansion_streak_length * 
                        volume_decrease_magnitude * 
                        range_contraction_magnitude)
        else:
            factor[i] = 0
    
    # Create output series
    factor_series = pd.Series(factor, index=df.index)
    
    return factor_series
