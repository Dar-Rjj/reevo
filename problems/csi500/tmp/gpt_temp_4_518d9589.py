import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Morning Gap Indicator
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    gap_threshold = data['gap'].quantile(0.8)
    
    # Intraday Price Range Normalization
    data['price_range'] = (data['high'] - data['low']) / data['open']
    avg_range = data['price_range'].expanding().mean()
    
    # Midpoint Calculation
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['close_to_mid'] = data['close'] / data['midpoint']
    
    # Volatility and Momentum Calculations
    data['intraday_vol'] = (data['high'] - data['low']) / data['open']
    data['rolling_mean_close'] = data['close'].expanding().mean()
    data['momentum'] = data['close'] - data['rolling_mean_close']
    data['adj_momentum'] = data['momentum'] / data['intraday_vol'].replace(0, np.nan)
    
    # Volume Calculations
    data['rolling_vol_ma'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['vol_ratio'] = data['volume'] / data['rolling_vol_ma']
    data['rolling_std_close'] = data['close'].rolling(window=5, min_periods=1).std()
    
    # Volume-Enhanced Momentum
    data['vol_enhanced_momentum'] = data['adj_momentum'] * data['volume']
    data['norm_momentum'] = data['vol_enhanced_momentum'] / data['rolling_std_close'].replace(0, np.nan)
    
    # Signal Generation
    for i in range(1, len(data)):
        current = data.iloc[i]
        prev_close = data.iloc[i-1]['close']
        
        # Check for overreaction conditions
        gap_condition = (current['gap'] > gap_threshold) or (current['gap'] < -gap_threshold)
        range_condition = current['price_range'] > avg_range.iloc[i]
        
        if gap_condition and range_condition:
            # Check for strong moves with volume confirmation
            if (current['high'] > prev_close * 1.02) and (current['vol_ratio'] > 1.5):
                factor.iloc[i] = -1  # Sell signal
            elif (current['low'] < prev_close * 0.98) and (current['vol_ratio'] > 1.5):
                factor.iloc[i] = 1   # Buy signal
            else:
                factor.iloc[i] = 0   # No signal
        else:
            factor.iloc[i] = 0       # No signal
    
    # Fill first day with 0 (no signal possible)
    factor.iloc[0] = 0
    
    return factor
