import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # High-Low Range Momentum Persistence
    # High-Low range change vs 5-day average
    high_low_range = data['high'] - data['low']
    range_5d_avg = high_low_range.rolling(window=5, min_periods=3).mean()
    range_change = high_low_range - range_5d_avg
    
    # Volume trend slope (5-day linear regression slope)
    volume_trend = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            window_vol = data['volume'].iloc[i-4:i+1]
            if len(window_vol) == 5:
                x = np.arange(5)
                slope = np.polyfit(x, window_vol.values, 1)[0]
                volume_trend.iloc[i] = slope
    
    hl_range_momentum = range_change * volume_trend
    
    # Open-Gap Momentum Divergence
    # Overnight gap vs intraday return direction
    prev_close = data['close'].shift(1)
    overnight_gap = (data['open'] - prev_close) / prev_close
    intraday_return = (data['close'] - data['open']) / data['open']
    
    # Gap magnitude * intraday momentum
    gap_momentum = np.abs(overnight_gap) * intraday_return
    
    # Volume-Range Breakout Detector
    # Range expansion vs 5-day average
    range_expansion = high_low_range / range_5d_avg - 1
    
    # Volume expansion vs 20-day average
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_expansion = data['volume'] / volume_20d_avg - 1
    
    volume_range_breakout = range_expansion * volume_expansion
    
    # Close-Range Volatility Persistence
    # Consecutive range expansion count
    range_expansion_count = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if high_low_range.iloc[i] > high_low_range.iloc[i-1]:
            range_expansion_count.iloc[i] = range_expansion_count.iloc[i-1] + 1
        else:
            range_expansion_count.iloc[i] = 0
    
    # Weight by current range magnitude
    range_persistence = range_expansion_count * high_low_range
    
    # Intraday Range Pressure Index
    # Buying pressure minus selling pressure
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    buying_pressure = np.where(data['close'] > data['open'], 
                              data['close'] - data['low'], 0)
    selling_pressure = np.where(data['close'] < data['open'], 
                               data['high'] - data['close'], 0)
    range_pressure = buying_pressure - selling_pressure
    
    # 5-day volume trend (simple moving average slope)
    volume_5d_trend = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_5d_trend_slope = volume_5d_trend.diff()
    
    intraday_pressure = range_pressure * volume_5d_trend_slope
    
    # Combine all factors with equal weights
    factors = [hl_range_momentum, gap_momentum, volume_range_breakout, 
               range_persistence, intraday_pressure]
    
    # Standardize each factor and combine
    for i in range(len(data)):
        if i >= 20:  # Ensure enough data for calculations
            factor_values = []
            for f in factors:
                if not pd.isna(f.iloc[i]):
                    # Z-score using rolling 20-day window
                    window_data = f.iloc[max(0, i-19):i+1]
                    if len(window_data) >= 10:
                        mean_val = window_data.mean()
                        std_val = window_data.std()
                        if std_val > 0:
                            z_score = (f.iloc[i] - mean_val) / std_val
                            factor_values.append(z_score)
            
            if factor_values:
                factor.iloc[i] = np.mean(factor_values)
    
    return factor
