import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate basic components
    data['AM_Momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['PM_Momentum'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Calculate 3-day rolling correlation between AM and PM momentum
    am_series = data['AM_Momentum']
    pm_series = data['PM_Momentum']
    
    # Create rolling correlation using expanding window approach
    rolling_corr = []
    for i in range(len(data)):
        if i < 2:  # Need at least 3 points for correlation
            rolling_corr.append(np.nan)
        else:
            window_am = am_series.iloc[i-2:i+1]
            window_pm = pm_series.iloc[i-2:i+1]
            if len(window_am) >= 2 and len(window_pm) >= 2:
                corr_val = window_am.corr(window_pm)
                rolling_corr.append(corr_val if not pd.isna(corr_val) else 0)
            else:
                rolling_corr.append(0)
    
    data['Momentum_Correlation'] = rolling_corr
    
    # Calculate momentum divergence signal
    data['Momentum_Divergence'] = data['Momentum_Correlation'] * abs(data['AM_Momentum'] - data['PM_Momentum'])
    
    # Calculate volume acceleration component
    data['Price_Volume_Interaction'] = (data['close'] - data['open']) * data['volume']
    
    # Calculate 3-day rolling mean of volume
    data['Volume_3day_MA'] = data['volume'].rolling(window=3, min_periods=1).mean()
    
    # Calculate 5-day rolling sum of price-volume interaction
    data['Price_Volume_5day_Sum'] = data['Price_Volume_Interaction'].rolling(window=5, min_periods=1).sum()
    
    # Calculate volume velocity
    data['Volume_Velocity'] = np.sign(data['Price_Volume_Interaction']) * (data['volume'] / data['Volume_3day_MA']) * data['Price_Volume_5day_Sum']
    
    # Calculate position shift component
    data['Position_Ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['Position_Shift'] = data['Position_Ratio'] - data['Position_Ratio'].shift(1)
    
    # Combine components to form final factor
    # Use momentum divergence and volume velocity as primary signals
    # Position shift acts as reversal confirmation
    data['Factor'] = data['Momentum_Divergence'] * data['Volume_Velocity'] * data['Position_Shift']
    
    # Handle any remaining NaN values
    data['Factor'] = data['Factor'].fillna(0)
    
    return data['Factor']
