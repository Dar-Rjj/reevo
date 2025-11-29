import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic metrics
    df['hl_range'] = df['high'] / df['low'] - 1
    df['oc_range'] = df['close'] / df['open'] - 1
    df['dollar_volume'] = df['volume'] * df['close']
    df['amount_per_volume'] = df['amount'] / df['volume'].replace(0, np.nan)
    
    # Multi-scale periods for fractal analysis
    periods = [1, 3, 5, 10]
    
    for i in range(max(periods), len(df)):
        current_data = df.iloc[:i+1]
        
        # Volume-Price Fractal Efficiency Components
        vpfe_components = []
        for period in periods:
            if i >= period:
                # Price change per unit volume
                price_change = (current_data['close'].iloc[i] / current_data['close'].iloc[i-period] - 1)
                volume_sum = current_data['volume'].iloc[i-period+1:i+1].sum()
                if volume_sum > 0:
                    efficiency = abs(price_change) / (volume_sum ** 0.5)
                    vpfe_components.append(efficiency)
        
        # Opening Auction Quality
        opening_components = []
        if i >= 1:
            # First hour efficiency (using previous day's data)
            prev_day = current_data.iloc[i-1]
            opening_range_eff = (prev_day['high'] - prev_day['low']) / prev_day['open']
            volume_concentration = prev_day['volume'] / current_data['volume'].iloc[max(0,i-5):i].mean()
            opening_components.append(opening_range_eff * volume_concentration)
        
        # Amount Flow Fractal Dynamics
        amount_components = []
        for period in [3, 5]:
            if i >= period:
                # Dollar volume patterns
                dollar_vol_change = current_data['dollar_volume'].iloc[i-period+1:i+1].std() / \
                                  current_data['dollar_volume'].iloc[i-period+1:i+1].mean()
                price_impact = abs(current_data['close'].iloc[i] / current_data['close'].iloc[i-period] - 1) / \
                             current_data['dollar_volume'].iloc[i-period+1:i+1].sum()
                amount_components.append(dollar_vol_change * price_impact)
        
        # Close-Relative Efficiency Positioning
        position_components = []
        if i >= 1:
            current_day = current_data.iloc[i]
            # Position within daily range
            if current_day['high'] != current_day['low']:
                position_eff = (current_day['close'] - current_day['low']) / (current_day['high'] - current_day['low'])
                # Volume-weighted position strength
                volume_weight = current_day['volume'] / current_data['volume'].iloc[max(0,i-10):i+1].mean()
                position_components.append(position_eff * volume_weight)
        
        # Combine all components with weights
        vpfe_score = np.mean(vpfe_components) if vpfe_components else 0
        opening_score = np.mean(opening_components) if opening_components else 0
        amount_score = np.mean(amount_components) if amount_components else 0
        position_score = np.mean(position_components) if position_components else 0
        
        # Final factor value - weighted combination
        factor_value = (0.4 * vpfe_score + 
                       0.25 * opening_score + 
                       0.2 * amount_score + 
                       0.15 * position_score)
        
        factor_values.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
