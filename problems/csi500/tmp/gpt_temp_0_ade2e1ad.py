import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Calculate daily range
    data['daily_range'] = data['high'] - data['low']
    
    # Calculate VWAP
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate various rolling statistics
    data['vol_10d_median'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['vol_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['vol_10d_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Calculate rolling standard deviations
    data['std_5d'] = data['returns'].rolling(window=5, min_periods=3).std()
    data['std_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Calculate returns over different periods
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    
    # Calculate volume slopes
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window and not np.isnan(y).any():
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['vol_slope_5d'] = calculate_slope(data['volume'], 5)
    data['vol_slope_10d'] = calculate_slope(data['volume'], 10)
    
    # Calculate opening hour volume (approximated as first hour volume)
    # Assuming first hour has higher volume, we use 25% of daily volume as proxy
    data['opening_volume'] = data['volume'] * 0.25
    
    # Calculate pressure sign and consecutive days
    data['pressure_sign'] = np.sign(data['close'] - data['vwap'])
    data['pressure_consecutive'] = 0
    current_sign = 0
    consecutive_count = 0
    
    for i in range(len(data)):
        if i == 0 or np.isnan(data['pressure_sign'].iloc[i]):
            consecutive_count = 1
            current_sign = data['pressure_sign'].iloc[i] if not np.isnan(data['pressure_sign'].iloc[i]) else 0
        elif data['pressure_sign'].iloc[i] == current_sign:
            consecutive_count += 1
        else:
            consecutive_count = 1
            current_sign = data['pressure_sign'].iloc[i]
        data['pressure_consecutive'].iloc[i] = consecutive_count
    
    # Calculate factors for each day
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[i]
        prev_close = data['close'].iloc[i-1] if i > 0 else np.nan
        
        # Skip if insufficient data
        if (np.isnan(prev_close) or 
            np.isnan(current_data['vol_10d_median']) or
            np.isnan(current_data['vol_5d_avg']) or
            np.isnan(current_data['vol_10d_avg']) or
            np.isnan(current_data['range_5d_avg']) or
            np.isnan(current_data['atr_10d']) or
            np.isnan(current_data['std_5d']) or
            np.isnan(current_data['std_20d']) or
            np.isnan(current_data['return_1d']) or
            np.isnan(current_data['return_3d']) or
            np.isnan(current_data['return_5d']) or
            np.isnan(current_data['vol_slope_5d']) or
            np.isnan(current_data['vol_slope_10d'])):
            factor.iloc[i] = 0
            continue
        
        # 1. Intraday Gap Absorption Factor
        gap_size = (current_data['open'] / prev_close) - 1
        abs_gap_size = abs(gap_size)
        
        if abs_gap_size > 0:
            if gap_size > 0:
                absorption_strength = (current_data['high'] - current_data['open']) / abs_gap_size
            else:
                absorption_strength = (current_data['open'] - current_data['low']) / abs_gap_size
        else:
            absorption_strength = 0
            
        volume_intensity = current_data['volume'] / current_data['vol_10d_median']
        gap_absorption_factor = absorption_strength * volume_intensity if abs_gap_size > 0 else 0
        
        # 2. Volatility Regime Transition Factor
        volatility_ratio = current_data['std_5d'] / current_data['std_20d'] if current_data['std_20d'] > 0 else 1
        price_efficiency = current_data['daily_range'] / current_data['atr_10d'] if current_data['atr_10d'] > 0 else 1
        volume_change = current_data['volume'] / current_data['vol_5d_avg']
        volatility_transition_factor = volatility_ratio * price_efficiency * volume_change
        
        # 3. Momentum Acceleration Factor
        if (abs(current_data['return_5d'] - current_data['return_3d']) > 0.001 and 
            not np.isnan(current_data['vol_slope_5d']) and 
            not np.isnan(current_data['vol_slope_10d']) and
            abs(current_data['vol_slope_10d']) > 0.001):
            price_acceleration = (current_data['return_3d'] - current_data['return_1d']) / (current_data['return_5d'] - current_data['return_3d'])
            volume_acceleration = current_data['vol_slope_5d'] / current_data['vol_slope_10d']
            momentum_acceleration_factor = price_acceleration * volume_acceleration
        else:
            momentum_acceleration_factor = 0
        
        # 4. Intraday Pressure Build-up Factor
        if current_data['daily_range'] > 0:
            pressure_index = (current_data['close'] - current_data['vwap']) / current_data['daily_range']
        else:
            pressure_index = 0
            
        pressure_duration = current_data['pressure_consecutive']
        volume_density = current_data['volume'] / current_data['daily_range'] if current_data['daily_range'] > 0 else 0
        pressure_buildup_factor = pressure_index * pressure_duration * volume_density
        
        # 5. Range Expansion Quality Factor
        range_expansion = current_data['daily_range'] / current_data['range_5d_avg'] if current_data['range_5d_avg'] > 0 else 1
        breakout_purity = abs(current_data['close'] - current_data['open']) / current_data['daily_range'] if current_data['daily_range'] > 0 else 0
        volume_profile = current_data['volume'] / current_data['opening_volume'] if current_data['opening_volume'] > 0 else 1
        range_expansion_factor = range_expansion * breakout_purity * volume_profile
        
        # Combine all factors with equal weights
        combined_factor = (gap_absorption_factor + 
                          volatility_transition_factor + 
                          momentum_acceleration_factor + 
                          pressure_buildup_factor + 
                          range_expansion_factor)
        
        factor.iloc[i] = combined_factor
    
    # Normalize the factor
    factor = (factor - factor.mean()) / factor.std() if factor.std() > 0 else factor
    
    return factor
