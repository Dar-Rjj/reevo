import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['session_range'] = (data['high'] - data['low']) / data['open']
    data['close_change'] = (data['close'] - data['open']) / data['open']
    data['session_efficiency'] = np.where(data['session_range'] != 0, 
                                        np.abs(data['close_change']) / data['session_range'], 0)
    
    # Intraday Volatility-Volume Efficiency Momentum
    data['directional_vol_ratio'] = np.where(data['open'] > data['low'], 
                                           (data['high'] - data['open']) / (data['open'] - data['low']), 1.0)
    data['vol_volume_efficiency'] = data['session_efficiency'] * data['volume'] / (data['session_range'] + 1e-8)
    
    # Price-Amount Velocity Gap Reversal
    data['amount_velocity'] = data['amount'].rolling(window=5).mean() / (data['amount'] + 1e-8)
    data['gap_velocity_divergence'] = data['overnight_gap'] * data['amount_velocity']
    
    # Range-Persistence Volume Confirmation
    data['range_persistence'] = data['session_range'].rolling(window=3).std()
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['persistence_volume_alignment'] = data['range_persistence'] * data['volume_confirmation']
    
    # Opening-Closing Efficiency Asymmetry
    data['volume_concentration'] = data['amount'] / (data['amount'].rolling(window=5).mean() + 1e-8)
    data['efficiency_volume_mismatch'] = data['session_efficiency'] - data['volume_concentration']
    
    # Volatility-Amount Pressure Momentum
    data['volatility_pressure'] = data['session_range'] * data['volume'] / (data['amount'] + 1e-8)
    data['pressure_response_efficiency'] = data['session_efficiency'] / (data['volatility_pressure'] + 1e-8)
    
    # Combine all components with weights
    for i in range(len(data)):
        if i < 5:  # Skip first few days for rolling calculations
            factor.iloc[i] = 0
            continue
            
        # Component 1: Intraday Volatility-Volume Efficiency Momentum
        comp1 = data['vol_volume_efficiency'].iloc[i] * np.log1p(data['directional_vol_ratio'].iloc[i])
        
        # Component 2: Price-Amount Velocity Gap Reversal (negative for reversal)
        comp2 = -data['gap_velocity_divergence'].iloc[i]
        
        # Component 3: Range-Persistence Volume Confirmation
        comp3 = data['persistence_volume_alignment'].iloc[i] * np.sign(data['close_change'].iloc[i])
        
        # Component 4: Opening-Closing Efficiency Asymmetry
        comp4 = data['efficiency_volume_mismatch'].iloc[i] * data['close_change'].iloc[i]
        
        # Component 5: Volatility-Amount Pressure Momentum
        comp5 = data['pressure_response_efficiency'].iloc[i] * data['volatility_pressure'].iloc[i]
        
        # Combine components with normalization
        raw_factor = (comp1 * 0.25 + comp2 * 0.20 + comp3 * 0.20 + comp4 * 0.15 + comp5 * 0.20)
        
        # Apply rolling normalization
        if i >= 20:
            recent_values = factor.iloc[i-20:i]
            if recent_values.std() > 0:
                factor.iloc[i] = (raw_factor - recent_values.mean()) / recent_values.std()
            else:
                factor.iloc[i] = raw_factor
        else:
            factor.iloc[i] = raw_factor
    
    # Final normalization
    if len(factor) > 20:
        factor = (factor - factor.rolling(window=20, min_periods=1).mean()) / factor.rolling(window=20, min_periods=1).std()
    
    return factor.fillna(0)
