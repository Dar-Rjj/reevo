import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate components for each day
    for i in range(1, len(data)):
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Morning Momentum (open to high)
        morning_momentum = (current['high'] - current['open']) / current['open']
        morning_direction = 1 if (prev['close'] - prev['open']) >= 0 else -1
        morning_component = morning_momentum * morning_direction
        
        # Afternoon Momentum (low to close)
        afternoon_momentum = (current['close'] - current['low']) / current['low']
        afternoon_direction = 1 if (prev['close'] - prev['open']) >= 0 else -1
        afternoon_component = afternoon_momentum * afternoon_direction
        
        # Intraday Trend Component
        intraday_trend = morning_component + afternoon_component
        
        # Momentum Adjustment (5-day rolling momentum)
        if i >= 5:
            daily_momentum = []
            for j in range(i-5, i):
                if j > 0:
                    daily_momentum.append(data.iloc[j]['close'] - data.iloc[j-1]['close'])
            rolling_momentum = sum(daily_momentum)
            
            # Normalize and apply EMA smoothing
            if rolling_momentum != 0:
                normalized = intraday_trend / rolling_momentum
            else:
                normalized = 0
                
            # 3-day EMA smoothing
            if i >= 7:
                ema_prev = factor.iloc[i-1]
                factor.iloc[i] = normalized * (2/4) + ema_prev * (2/4)
            else:
                factor.iloc[i] = normalized
        else:
            factor.iloc[i] = intraday_trend
    
    return factor
