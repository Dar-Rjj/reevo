import pandas as pd
import numpy as np
def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Compute components
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        
        # Morning Momentum: (Open - Close_prev) / Close_prev
        morning_momentum = (current['open'] - previous['close']) / previous['close']
        
        # Afternoon Momentum: (Close - Midday Price) / Midday Price
        midday_price = (current['high'] + current['low']) / 2
        afternoon_momentum = (current['close'] - midday_price) / midday_price
        
        # Intraday Momentum (average of morning and afternoon)
        intraday_momentum = (morning_momentum + afternoon_momentum) / 2
        
        # Volume Trend: current volume / 5-day moving average volume (using only past data)
        if i >= 5:
            vol_ma = df.iloc[i-5:i]['volume'].mean()
            volume_trend = current['volume'] / vol_ma
        else:
            volume_trend = 1  # neutral value when not enough history
            
        # Scale Momentum by Volume Trend
        scaled_momentum = intraday_momentum * volume_trend
        
        # Normalize using historical standard deviation (20 days)
        if i >= 20:
            # Calculate standard deviation using only past data
            past_momentums = []
            for j in range(max(0, i-20), i):
                prev_current = df.iloc[j]
                prev_previous = df.iloc[j-1] if j > 0 else df.iloc[j]
                
                # Calculate past momentum components
                prev_morning = (prev_current['open'] - prev_previous['close']) / prev_previous['close']
                prev_midday = (prev_current['high'] + prev_current['low']) / 2
                prev_afternoon = (prev_current['close'] - prev_midday) / prev_midday
                prev_intraday = (prev_morning + prev_afternoon) / 2
                
                # Calculate past volume trend
                if j >= 5:
                    prev_vol_ma = df.iloc[j-5:j]['volume'].mean()
                    prev_vol_trend = prev_current['volume'] / prev_vol_ma
                else:
                    prev_vol_trend = 1
                
                past_scaled_momentum = prev_intraday * prev_vol_trend
                past_momentums.append(past_scaled_momentum)
            
            std_dev = np.std(past_momentums)
            if std_dev != 0:
                normalized_momentum = scaled_momentum / std_dev
            else:
                normalized_momentum = scaled_momentum
        else:
            normalized_momentum = scaled_momentum
            
        factor.iloc[i] = normalized_momentum
    
    return factor
