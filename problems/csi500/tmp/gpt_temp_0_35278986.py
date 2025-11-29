import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Intraday Range-Price Divergence Analysis
    # Range Efficiency Patterns
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_per_range'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['range_efficiency_divergence'] = data['intraday_efficiency'] * data['volume_per_range']
    
    # Price-Volume Directional Alignment
    data['price_direction'] = np.sign(data['close'] - data['open'])
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['directional_divergence'] = data['price_direction'] * data['volume_direction']
    
    # Multi-Timeframe Divergence Confirmation
    # Short-term (5-day) divergence
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['short_term_divergence'] = data['price_momentum_5d'] * data['volume_momentum_5d']
    
    # Medium-term (20-day) divergence
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['medium_term_divergence'] = data['price_momentum_20d'] * data['volume_momentum_20d']
    
    # Divergence Persistence and Regime Detection
    # Directional persistence (3-day)
    data['directional_persistence'] = 0
    for i in range(2, len(data)):
        if i >= 2:
            current_sign = data['directional_divergence'].iloc[i]
            prev_sign_1 = data['directional_divergence'].iloc[i-1]
            prev_sign_2 = data['directional_divergence'].iloc[i-2]
            
            if current_sign == prev_sign_1 == prev_sign_2:
                data.loc[data.index[i], 'directional_persistence'] = 3
            elif current_sign == prev_sign_1:
                data.loc[data.index[i], 'directional_persistence'] = 2
            else:
                data.loc[data.index[i], 'directional_persistence'] = 1
    
    # Magnitude persistence (3-day average)
    data['magnitude_persistence'] = data['range_efficiency_divergence'].rolling(window=3, min_periods=1).mean()
    data['persistence_score'] = data['directional_persistence'] * data['magnitude_persistence']
    
    # Regime Detection
    data['timeframe_alignment'] = np.sign(data['short_term_divergence']) * np.sign(data['medium_term_divergence'])
    data['divergence_trend'] = data['short_term_divergence'] - data['medium_term_divergence']
    data['regime_quality'] = data['timeframe_alignment'] * data['divergence_trend']
    
    # Final Composite Factor Construction
    # Core divergence signal integration
    data['intraday_composite'] = data['range_efficiency_divergence'] * data['directional_divergence']
    data['confirmed_divergence'] = data['intraday_composite'] * data['short_term_divergence'] * data['medium_term_divergence']
    
    # Persistence and regime enhancement
    data['persistence_enhanced'] = data['confirmed_divergence'] * data['persistence_score']
    data['final_factor'] = data['persistence_enhanced'] * data['regime_quality']
    
    # Return the final factor series
    return data['final_factor']
