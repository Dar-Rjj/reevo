import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate series
    data['prev_close'] = data['close'].shift(1)
    data['close_3d_ago'] = data['close'].shift(3)
    data['close_2d_ago'] = data['close'].shift(2)
    data['volume_2d_ago'] = data['volume'].shift(2)
    data['volume_1d_ago'] = data['volume'].shift(1)
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['prev_open'] = data['open'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['range_3d_avg'] = (data['high'] - data['low']).rolling(window=3, min_periods=1).mean()
    data['close_3d_ma'] = data['close'].rolling(window=3, min_periods=1).mean()
    
    # Calculate gap sign and consecutive counts
    data['gap'] = data['open'] / data['prev_close'] - 1
    data['gap_sign'] = np.sign(data['gap'])
    data['gap_sign_change'] = data['gap_sign'] != data['gap_sign'].shift(1)
    data['consecutive_count'] = data.groupby(data['gap_sign_change'].cumsum())['gap_sign'].cumcount() + 1
    
    for i, row in data.iterrows():
        if pd.isna(row['prev_close']) or pd.isna(row['close_3d_ago']) or pd.isna(row['close_2d_ago']):
            factor.loc[i] = np.nan
            continue
            
        # Factor 1: Intraday Momentum Reversal with Volatility Adjustment
        intraday_momentum = row['close'] - row['open']
        high_low_vol = row['high'] - row['low']
        if high_low_vol == 0:
            vol_adj_momentum = 0
        else:
            vol_adj_momentum = intraday_momentum / high_low_vol
        
        prev_day_return = row['close'] / row['prev_close'] - 1
        momentum_3d = row['close'] / row['close_3d_ago'] - 1
        
        factor1 = vol_adj_momentum * (prev_day_return + momentum_3d)
        
        # Factor 2: Price-Volume Velocity Divergence
        price_velocity = (row['close'] - row['close_2d_ago']) / 2
        volume_velocity = (row['volume'] - row['volume_2d_ago']) / 2
        
        if volume_velocity == 0:
            velocity_divergence = 0
        else:
            velocity_divergence = price_velocity / volume_velocity
        
        if (row['high'] - row['low']) == 0:
            range_efficiency = 0
        else:
            range_efficiency = (row['close'] - row['open']) / (row['high'] - row['low'])
        
        factor2 = velocity_divergence * range_efficiency
        
        # Factor 3: Opening Gap Persistence with Volume Confirmation
        opening_gap = row['open'] / row['prev_close']
        if row['volume_5d_avg'] == 0:
            volume_intensity = 0
        else:
            volume_intensity = row['volume'] / row['volume_5d_avg']
        
        gap_persistence = row['consecutive_count']
        
        factor3 = opening_gap * volume_intensity * gap_persistence
        
        # Factor 4: Intraday Pressure Accumulation
        if (row['high'] - row['low']) == 0:
            morning_pressure = 0
            afternoon_pressure = 0
        else:
            morning_pressure = (row['high'] - row['open']) / (row['high'] - row['low'])
            afternoon_pressure = (row['close'] - row['low']) / (row['high'] - row['low'])
        
        pressure_accumulation = morning_pressure * afternoon_pressure
        
        if row['volume_1d_ago'] == 0:
            volume_acceleration = 0
        else:
            volume_acceleration = row['volume'] / row['volume_1d_ago']
        
        factor4 = pressure_accumulation * volume_acceleration
        
        # Factor 5: Range Breakout with Momentum
        current_range = row['high'] - row['low']
        if row['range_3d_avg'] == 0:
            breakout_magnitude = 0
        else:
            breakout_magnitude = current_range / row['range_3d_avg']
        
        intraday_momentum_5 = row['close'] - row['open']
        if current_range == 0:
            vol_adj_momentum_5 = 0
        else:
            vol_adj_momentum_5 = intraday_momentum_5 / current_range
        
        factor5 = breakout_magnitude * vol_adj_momentum_5
        
        # Factor 6: Intraday Asymmetry with Consistency
        upper_range = row['high'] - row['open']
        lower_range = row['open'] - row['low']
        
        if lower_range == 0:
            asymmetry_ratio = 0
        else:
            asymmetry_ratio = upper_range / lower_range
        
        if (row['high'] - row['low']) == 0:
            range_efficiency_6 = 0
        else:
            range_efficiency_6 = (row['close'] - row['open']) / (row['high'] - row['low'])
        
        if row['close_3d_ma'] == 0:
            price_consistency = 0
        else:
            price_consistency = row['close'] / row['close_3d_ma']
        
        factor6 = asymmetry_ratio * range_efficiency_6 * price_consistency
        
        # Combine factors (equal weighting)
        combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5 + factor6) / 6
        factor.loc[i] = combined_factor
    
    return factor
