import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['price_range'] = data['high'] - data['low']
    data['daily_return'] = data['close'] - data['open']
    
    # Calculate volume and amount ratios
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    
    # Net Momentum Flow
    data['net_momentum_flow'] = (data['high'] - data['open']) * data['volume_amount_ratio'] - \
                               (data['open'] - data['low']) * data['volume_amount_ratio']
    
    # Momentum Efficiency Ratio
    data['momentum_efficiency_ratio'] = (data['volume_amount_ratio'] * data['daily_return']) / \
                                       (data['amount_volume_ratio'] * data['daily_return'])
    data['momentum_efficiency_ratio'] = data['momentum_efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Opening Momentum Efficiency (using first hour approximation)
    data['opening_momentum_efficiency'] = (data['high'] - data['open']) * data['volume_amount_ratio']
    
    # Closing Momentum Efficiency (using last hour approximation)
    data['closing_momentum_efficiency'] = (data['close'] - data['low']) * data['volume_amount_ratio']
    
    # Session Momentum Ratio
    data['session_momentum_ratio'] = data['opening_momentum_efficiency'] / data['closing_momentum_efficiency']
    data['session_momentum_ratio'] = data['session_momentum_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-Timeframe Momentum Persistence
    data['momentum_trend'] = data['net_momentum_flow'] - data['net_momentum_flow'].shift(3)
    
    # Momentum Flow Change and Acceleration
    data['momentum_flow_change'] = data['net_momentum_flow'] - data['net_momentum_flow'].shift(1)
    data['momentum_acceleration'] = data['net_momentum_flow'] - 2 * data['net_momentum_flow'].shift(1) + data['net_momentum_flow'].shift(2)
    
    # Acceleration Pattern
    data['acceleration_pattern'] = np.sign(data['momentum_flow_change']) * np.sign(data['momentum_acceleration'])
    
    # Range-based momentum components
    data['range_position_momentum'] = data['daily_return'] / data['price_range'] * data['volume']
    data['range_expansion_momentum'] = data['daily_return'] / data['price_range'] * data['volume']
    
    # Opening and Closing Range Momentum
    data['opening_range_momentum'] = (data['high'] - data['low']) * data['volume_amount_ratio']
    data['closing_range_momentum'] = (data['high'] - data['low']) * data['volume_amount_ratio']
    
    # Session Range Asymmetry
    data['session_range_asymmetry'] = data['opening_range_momentum'] / data['closing_range_momentum']
    data['session_range_asymmetry'] = data['session_range_asymmetry'].replace([np.inf, -np.inf], np.nan)
    
    # Directional Asymmetry
    data['directional_asymmetry'] = np.sign(data['daily_return']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Efficiency-Momentum Convergence
    data['efficiency_momentum_convergence'] = data['daily_return'] * data['momentum_efficiency_ratio']
    
    # Asymmetry Intensity
    data['asymmetry_intensity'] = np.abs(data['efficiency_momentum_convergence'])
    
    # Session Flow Divergence
    data['session_flow_divergence'] = data['opening_momentum_efficiency'] - data['closing_momentum_efficiency']
    
    # Deceleration Ratio
    data['deceleration_ratio'] = data['closing_momentum_efficiency'] / data['opening_momentum_efficiency']
    data['deceleration_ratio'] = data['deceleration_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Range Expansion Ratio
    data['range_expansion_ratio'] = data['price_range'] / (data['high'].shift(3) - data['low'].shift(3))
    data['range_expansion_ratio'] = data['range_expansion_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Range Expansion Momentum
    data['range_expansion_momentum'] = data['price_range'] - (data['high'].shift(1) - data['low'].shift(1))
    
    # Calculate rolling correlations for momentum-price divergence
    momentum_price_corr = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            if len(window_data) >= 3:
                corr = window_data['net_momentum_flow'].corr(window_data['daily_return'])
                momentum_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_price_corr.append(0)
        else:
            momentum_price_corr.append(0)
    
    data['momentum_price_divergence'] = momentum_price_corr
    
    # Calculate regime stability (consecutive days with same momentum direction)
    momentum_direction = np.sign(data['net_momentum_flow'])
    regime_stability = []
    current_streak = 0
    
    for i in range(len(data)):
        if i == 0:
            current_streak = 1
        else:
            if momentum_direction.iloc[i] == momentum_direction.iloc[i-1] and not pd.isna(momentum_direction.iloc[i]) and not pd.isna(momentum_direction.iloc[i-1]):
                current_streak += 1
            else:
                current_streak = 1
        regime_stability.append(current_streak)
    
    data['regime_stability'] = regime_stability
    
    # Session Asymmetry Score
    data['session_asymmetry_score'] = data['opening_momentum_efficiency'] * data['closing_momentum_efficiency'] * data['regime_stability']
    
    # Final composite factor calculation
    # Asymmetry Quality
    data['asymmetry_quality'] = data['momentum_price_divergence'] * data['efficiency_momentum_convergence'] * data['session_asymmetry_score']
    
    # Flow Quality
    data['flow_quality'] = data['acceleration_pattern'] * data['session_flow_divergence'] * data['deceleration_ratio']
    
    # Range Quality
    data['range_quality'] = data['range_expansion_momentum'] * data['momentum_efficiency_ratio'] * data['session_range_asymmetry']
    
    # Final composite factor - weighted combination of quality factors
    final_factor = (0.4 * data['asymmetry_quality'] + 
                   0.35 * data['flow_quality'] + 
                   0.25 * data['range_quality'])
    
    # Clean infinite values
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    
    return final_factor
