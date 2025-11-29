import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate overnight gap
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Calculate rolling median absolute deviation for gaps (15-day window)
    data['gap_mad'] = data['gap'].rolling(window=15, min_periods=10).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=False
    )
    
    # Identify extreme gaps (|gap| > 2.0 * MAD)
    data['extreme_gap'] = np.abs(data['gap']) > (2.0 * data['gap_mad'])
    
    # Calculate volume momentum (current volume / 5-day volume average excluding current)
    data['volume_ma_5'] = data['volume'].shift(1).rolling(window=5, min_periods=3).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma_5']
    
    # Calculate intraday velocity (High - Low) / (Amount / Volume)
    data['intraday_velocity'] = (data['high'] - data['low']) / (data['amount'] / data['volume'])
    data['intraday_velocity'] = data['intraday_velocity'].replace([np.inf, -np.inf], np.nan)
    
    # Combine volume momentum and intraday velocity
    data['volume_acceleration'] = data['volume_momentum'] * data['intraday_velocity']
    
    # Generate composite factor
    # Apply only when extreme gap detected, use mean reversion logic
    data['factor'] = np.nan
    extreme_mask = data['extreme_gap'] == True
    data.loc[extreme_mask, 'factor'] = -np.sign(data.loc[extreme_mask, 'gap']) * data.loc[extreme_mask, 'volume_acceleration']
    
    return data['factor']
