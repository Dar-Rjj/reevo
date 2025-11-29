import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Intraday Momentum Dynamics
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_acceleration'] = data['intraday_momentum'] - data['intraday_momentum'].shift(1)
    data['acceleration_direction'] = np.sign(data['momentum_acceleration'])
    
    # Acceleration Strength Quantification
    data['acceleration_magnitude'] = np.abs(data['momentum_acceleration'])
    data['price_efficiency_norm'] = np.abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['base_acceleration_signal'] = data['acceleration_direction'] * data['acceleration_magnitude'] * data['price_efficiency_norm']
    
    # Price-Volume Efficiency Analysis
    data['efficiency_direction'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['efficiency_magnitude'] = (np.abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)) * np.abs(data['volume'] - data['volume'].shift(1))
    data['efficiency_persistence'] = data['efficiency_direction'].rolling(window=3, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
    data['extreme_efficiency'] = (data['efficiency_magnitude'] > 2 * data['efficiency_magnitude'].rolling(window=10, min_periods=1).median()).astype(float)
    
    # Momentum Continuation Confirmation
    data['typical_price'] = (data['open'] + data['high'] + data['low']) / 3
    data['intraday_momentum_trend'] = np.sign(data['close'] - data['typical_price'])
    
    # Calculate momentum persistence using rolling window
    momentum_persistence_values = []
    for i in range(len(data)):
        if i < 2:
            momentum_persistence_values.append(0)
        else:
            window_data = data.iloc[max(0, i-2):i+1]
            persistence = (window_data['intraday_momentum_trend'] * window_data['intraday_momentum_trend'].shift(1)).sum()
            momentum_persistence_values.append(persistence)
    data['momentum_persistence'] = momentum_persistence_values
    
    data['absolute_momentum'] = np.abs(data['close'] - data['typical_price'])
    data['normalized_momentum'] = data['absolute_momentum'] / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_continuation_signal'] = data['normalized_momentum'] * np.abs(data['momentum_persistence'])
    
    # Volume Efficiency Integration
    data['absolute_volume_movement'] = np.abs(data['volume'] - data['volume'].shift(1))
    data['movement_per_unit_price'] = data['absolute_volume_movement'] / np.abs(data['close'] - data['open']).replace(0, np.nan)
    data['volume_efficiency_ratio'] = data['movement_per_unit_price'] / data['movement_per_unit_price'].rolling(window=5, min_periods=1).mean()
    
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1).replace(0, np.nan)) - 1
    data['volume_trend'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_acceleration_signal'] = data['volume_acceleration'] * data['volume_trend']
    
    # Signal Combination Layer
    data['efficiency_confirmed_acceleration'] = data['base_acceleration_signal'] * data['efficiency_direction']
    data['volume_adjusted_acceleration'] = data['efficiency_confirmed_acceleration'] * data['volume_efficiency_ratio']
    
    data['acceleration_based_continuation'] = data['base_acceleration_signal'] * data['momentum_continuation_signal']
    data['volume_acceleration_continuation'] = data['acceleration_based_continuation'] * data['volume_acceleration_signal']
    
    data['persistence_weighted_efficiency'] = data['efficiency_direction'] * (1 + data['efficiency_persistence'] / 3)
    data['extreme_efficiency_signal'] = data['persistence_weighted_efficiency'] * data['extreme_efficiency']
    data['efficiency_enhanced_acceleration'] = data['volume_adjusted_acceleration'] * data['extreme_efficiency_signal']
    
    # Multi-Timeframe Context
    data['short_term_momentum'] = data['base_acceleration_signal'].rolling(window=3, min_periods=1).sum()
    
    # Calculate rolling correlation
    def rolling_correlation(x):
        if len(x) < 2:
            return 0
        return np.corrcoef(np.abs(x['close'] - x['open']), x['volume'])[0, 1]
    
    medium_term_efficiency = []
    for i in range(len(data)):
        if i < 4:
            medium_term_efficiency.append(0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            corr = rolling_correlation(window_data[['open', 'close', 'volume']])
            medium_term_efficiency.append(corr if not np.isnan(corr) else 0)
    data['medium_term_efficiency'] = medium_term_efficiency
    
    data['multi_timeframe_confirmation'] = data['short_term_momentum'] * data['medium_term_efficiency']
    
    # Volatility Context Integration
    data['momentum_std_5d'] = data['intraday_momentum'].rolling(window=5, min_periods=1).std()
    data['momentum_ratio'] = np.abs(data['intraday_momentum']) / data['momentum_std_5d'].replace(0, np.nan)
    data['extreme_momentum_condition'] = (data['momentum_ratio'] > 1).astype(float)
    data['volatility_factor'] = 1 / data['momentum_ratio'].replace(0, np.nan)
    
    # Final Factor Construction
    data['core_factor'] = data['efficiency_enhanced_acceleration'] * data['volume_acceleration_continuation']
    data['confirmation_factor'] = data['core_factor'] * data['multi_timeframe_confirmation']
    data['volatility_adjusted_factor'] = data['confirmation_factor'] * data['volatility_factor']
    
    data['persistence_weighting'] = 1 + data['efficiency_persistence'] / 3
    data['momentum_consistency'] = data['intraday_momentum_trend'].rolling(window=3, min_periods=1).mean()
    data['final_factor'] = data['volatility_adjusted_factor'] * data['persistence_weighting'] * data['momentum_consistency']
    
    # Output Enhancement
    data['noise_reduction'] = data['final_factor'].rolling(window=3, min_periods=1).median()
    data['signal_strength'] = data['final_factor'] * np.abs(data['volume_acceleration_signal'])
    data['enhanced_alpha'] = data['noise_reduction'] * data['signal_strength']
    
    return data['enhanced_alpha']
