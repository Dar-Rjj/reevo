import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Fractal Price Movement Analysis
    data['intraday_oscillation'] = (data['high'] - data['low']) / data['open']
    
    # Calculate directional changes per day (price path complexity)
    data['price_changes'] = 0
    for i in range(1, len(data)):
        prev_close = data['close'].iloc[i-1]
        curr_open = data['open'].iloc[i]
        curr_high = data['high'].iloc[i]
        curr_low = data['low'].iloc[i]
        curr_close = data['close'].iloc[i]
        
        changes = 0
        # Open vs previous close
        if curr_open > prev_close:
            changes += 1
        elif curr_open < prev_close:
            changes += 1
            
        # High vs open
        if curr_high > curr_open:
            changes += 1
        elif curr_high < curr_open:
            changes += 1
            
        # Low vs open  
        if curr_low < curr_open:
            changes += 1
        elif curr_low > curr_open:
            changes += 1
            
        # Close vs open
        if curr_close > curr_open:
            changes += 1
        elif curr_close < curr_open:
            changes += 1
            
        data.iloc[i, data.columns.get_loc('price_changes')] = changes
    
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Distribution Skew Patterns
    data['mid_price'] = (data['high'] + data['low']) / 2
    # Estimate upper volume (assuming symmetric distribution around mid-price)
    data['upper_volume_est'] = data['volume'] * (data['close'] - data['mid_price']) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_volume_est'] = data['volume'] - data['upper_volume_est']
    data['volume_skew'] = data['upper_volume_est'] / data['lower_volume_est'].replace(0, np.nan)
    
    # Price-Volume Temporal Decoupling
    data['prev_close'] = data['close'].shift(1)
    # Estimate morning session (first half of trading day)
    data['morning_momentum'] = (data['open'] - data['prev_close']) * (data['volume'] * 0.4)  # Assume 40% volume in morning
    # Estimate midday price as average of open and (high+low)/2
    data['midday_price'] = (data['open'] + data['mid_price']) / 2
    # Estimate afternoon session
    data['afternoon_momentum'] = (data['close'] - data['midday_price']) * (data['volume'] * 0.6)  # Assume 60% volume in afternoon
    data['session_divergence'] = data['morning_momentum'] - data['afternoon_momentum']
    
    # Amount-Price Elasticity Framework
    data['amount_per_unit'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    # Calculate elasticity as change in amount relative to change in price range
    data['amount_elasticity'] = data['amount_per_unit'].pct_change() / data['intraday_oscillation'].pct_change().replace(0, np.nan)
    
    # Multi-Scale Fractal Integration
    # Combine components with appropriate weights
    for i in range(len(data)):
        if i < 1:  # Need at least 1 day of history
            factor_values.iloc[i] = 0
            continue
            
        # Normalize components
        oscillation_norm = data['intraday_oscillation'].iloc[i] / data['intraday_oscillation'].iloc[:i+1].mean()
        efficiency_norm = data['price_efficiency'].iloc[i] if not pd.isna(data['price_efficiency'].iloc[i]) else 0
        skew_norm = data['volume_skew'].iloc[i] if not pd.isna(data['volume_skew'].iloc[i]) else 1
        divergence_norm = data['session_divergence'].iloc[i] / abs(data['session_divergence'].iloc[:i+1]).mean() if abs(data['session_divergence'].iloc[:i+1]).mean() > 0 else 0
        elasticity_norm = data['amount_elasticity'].iloc[i] if not pd.isna(data['amount_elasticity'].iloc[i]) else 0
        
        # Integrated factor calculation
        factor = (oscillation_norm * 0.2 + 
                 efficiency_norm * 0.25 + 
                 np.log(skew_norm) * 0.15 + 
                 divergence_norm * 0.2 + 
                 np.tanh(elasticity_norm) * 0.2)
        
        factor_values.iloc[i] = factor
    
    # Fill any remaining NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
