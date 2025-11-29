import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Persistence Framework
    # Multi-timeframe Momentum Strength
    data['momentum_2d'] = data['close'] / data['close'].shift(2)
    data['momentum_2d_lagged'] = data['close'].shift(1) / data['close'].shift(3)
    data['momentum_2d_persistence'] = data['momentum_2d'] * data['momentum_2d_lagged']
    
    data['momentum_5d'] = data['close'] / data['close'].shift(5)
    data['momentum_5d_lagged'] = data['close'].shift(1) / data['close'].shift(6)
    data['momentum_acceleration'] = data['momentum_5d'] - data['momentum_5d_lagged']
    
    data['momentum_consistency'] = np.sign(data['momentum_2d']) * np.sign(data['momentum_5d'])
    
    # Momentum Regime Detection
    data['daily_returns'] = data['close'].pct_change()
    data['rolling_momentum_vol'] = data['daily_returns'].rolling(window=10, min_periods=5).std()
    data['current_momentum_magnitude'] = abs(data['close'] / data['close'].shift(1) - 1)
    data['momentum_regime_flag'] = (data['current_momentum_magnitude'] > 
                                   (1.5 * data['rolling_momentum_vol'])).astype(int)
    
    # Volume-Price Efficiency System
    # Price Movement Efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['overnight_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['total_efficiency_score'] = data['intraday_efficiency'] * data['overnight_efficiency']
    
    # Volume Confirmation Metrics
    data['volume_3d_median'] = data['volume'].rolling(window=3, min_periods=2).median()
    data['volume_persistence'] = data['volume'] / data['volume_3d_median']
    data['volume_direction_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['efficiency_volume_score'] = (data['total_efficiency_score'] * 
                                      data['volume_persistence'] * 
                                      data['volume_direction_alignment'])
    
    # Volatility Regime Switching Engine
    # Volatility State Classification
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['current_volatility'] = data['true_range'] / data['close']
    data['regime_threshold'] = data['current_volatility'].rolling(window=10, min_periods=5).median()
    data['high_volatility_flag'] = (data['current_volatility'] > 
                                   (1.8 * data['regime_threshold'])).astype(int)
    
    # Regime-Adaptive Parameters
    data['regime_multiplier'] = np.where(data['high_volatility_flag'] == 1, 1.5, 1.0)
    
    # Calculate regime persistence (consecutive days in current regime)
    data['regime_persistence'] = 0
    for i in range(1, len(data)):
        if data['high_volatility_flag'].iloc[i] == data['high_volatility_flag'].iloc[i-1]:
            data['regime_persistence'].iloc[i] = data['regime_persistence'].iloc[i-1] + 1
        else:
            data['regime_persistence'].iloc[i] = 1
    
    # Cross-Sectional Momentum Integration
    # Relative Momentum Strength
    data['momentum_5d'] = data['close'] / data['close'].shift(5)
    data['momentum_20d'] = data['close'] / data['close'].shift(20)
    data['momentum_10d'] = data['close'] / data['close'].shift(10)
    
    data['momentum_ratio'] = data['momentum_5d'] / data['momentum_20d']
    data['momentum_trend'] = data['momentum_5d'] / data['momentum_10d']
    data['momentum_stability'] = 1 - abs(data['momentum_5d'] - data['momentum_10d']) / data['momentum_5d']
    
    # Volume-Momentum Divergence
    data['price_volume_divergence'] = (data['close'] - data['open']) * (data['volume'] - data['volume'].shift(1))
    data['momentum_volume_alignment'] = data['momentum_5d'] * data['volume_persistence']
    data['divergence_score'] = data['price_volume_divergence'] * data['momentum_volume_alignment']
    
    # Signal Construction & Regime Application
    # Core Momentum Signal
    data['base_momentum'] = data['momentum_2d_persistence'] * data['momentum_consistency']
    data['enhanced_momentum'] = data['base_momentum'] * data['momentum_acceleration']
    data['regime_adjusted_momentum'] = data['enhanced_momentum'] * data['regime_multiplier']
    
    # Efficiency-Volume Signal
    data['core_efficiency'] = data['total_efficiency_score'] * data['volume_direction_alignment']
    data['volume_weighted_efficiency'] = data['core_efficiency'] * data['volume_persistence']
    data['regime_efficiency'] = data['volume_weighted_efficiency'] * data['regime_persistence']
    
    # Cross-Sectional Enhancement
    data['relative_momentum_component'] = data['momentum_trend'] * data['momentum_stability']
    data['divergence_adjustment'] = data['relative_momentum_component'] * data['divergence_score']
    data['cross_sectional_signal'] = data['divergence_adjustment'] * data['regime_multiplier']
    
    # Final Factor Generation
    # Signal Integration
    data['primary_factor'] = data['regime_adjusted_momentum'] * data['regime_efficiency']
    data['secondary_factor'] = data['cross_sectional_signal'] * data['volume_weighted_efficiency']
    data['tertiary_adjustment'] = data['primary_factor'] * data['secondary_factor']
    
    # Regime-Based Weighting
    data['final_alpha_factor'] = np.where(
        data['high_volatility_flag'] == 1,
        data['primary_factor'] * 1.2,
        data['secondary_factor'] * 0.8
    )
    
    # Handle transition phases (when regime changes)
    regime_changes = data['high_volatility_flag'].diff().fillna(0) != 0
    data.loc[regime_changes, 'final_alpha_factor'] = (
        data.loc[regime_changes, 'primary_factor'] + 
        data.loc[regime_changes, 'secondary_factor']
    ) / 2
    
    return data['final_alpha_factor']
