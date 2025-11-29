import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Efficiency Analysis
    # Intraday Volatility Patterns
    data['volatility_t'] = data['high'] - data['low']
    data['volatility_t_1'] = data['volatility_t'].shift(1)
    data['volatility_compression_efficiency'] = data['volatility_t'] / data['volatility_t_1']
    data['regime_persistence'] = np.sign(data['volatility_t'] - data['volatility_t_1'])
    data['volatility_momentum'] = data['regime_persistence'] * data['volatility_compression_efficiency']
    
    # Multi-Timeframe Volatility Alignment
    data['volatility_t_3'] = data['volatility_t'].shift(3)
    data['short_term_vol_ratio'] = data['volatility_t'] / data['volatility_t_3']
    data['volatility_convergence'] = data['volatility_compression_efficiency'] * data['short_term_vol_ratio']
    data['regime_transition_signals'] = data['volatility_convergence'] * data['regime_persistence']
    
    # Volatility Efficiency Metrics
    data['price_movement_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volatility_adjusted_efficiency'] = data['price_movement_efficiency'] * data['volatility_compression_efficiency']
    data['efficiency_momentum'] = data['volatility_adjusted_efficiency'] / data['volatility_adjusted_efficiency'].shift(1)
    
    # Liquidity Acceleration Framework
    # Volume Dynamics Analysis
    data['amount_t_1'] = data['amount'].shift(1)
    data['amount_t_2'] = data['amount'].shift(2)
    data['volume_velocity'] = data['amount'] / data['amount_t_1']
    data['volume_acceleration'] = (data['amount'] - data['amount_t_1']) / data['amount_t_2']
    
    # Calculate volume persistence (consecutive days with volume velocity > 1.2)
    data['volume_velocity_gt_1_2'] = (data['volume_velocity'] > 1.2).astype(int)
    data['volume_persistence'] = data['volume_velocity_gt_1_2'].groupby(data.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if (x == 1).any() else 0, raw=False
    ).reset_index(level=0, drop=True)
    
    # Liquidity Distribution Patterns (simplified using daily data)
    data['volume_concentration'] = 1.0  # Placeholder for hourly concentration
    data['early_late_volume_ratio'] = 1.0  # Placeholder for hourly ratio
    data['distribution_skew'] = data['volume_concentration'] * data['early_late_volume_ratio']
    
    # Liquidity Momentum Signals
    data['volume_weighted_momentum'] = (data['close'] - data['open']) * data['amount']
    data['liquidity_acceleration_factor'] = data['volume_weighted_momentum'] * data['volume_acceleration']
    data['volume_exhaustion'] = data['volume_velocity'] * abs(data['close'] - data['open'])
    
    # Regime-Adaptive Momentum Construction
    # Volatility-Regime Specific Momentum
    data['high_volatility_momentum'] = abs(data['close'] - data['open']) * data['volatility_compression_efficiency']
    data['low_volatility_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['regime_transition_momentum'] = data['high_volatility_momentum'] * data['low_volatility_momentum']
    
    # Liquidity-Constrained Momentum Factors
    data['volume_constrained_return'] = (data['close'] - data['open']) / data['volume_velocity']
    data['acceleration_adjusted_momentum'] = data['volume_constrained_return'] * data['volume_acceleration']
    data['persistence_weighted_signal'] = data['acceleration_adjusted_momentum'] * data['volume_persistence']
    
    # Multi-Timeframe Momentum Integration
    data['close_open_t_1'] = (data['close'] - data['open']).shift(1)
    data['short_term_momentum_alignment'] = (data['close'] - data['open']) * data['close_open_t_1']
    data['volatility_momentum_convergence'] = data['short_term_momentum_alignment'] * data['volatility_convergence']
    data['liquidity_momentum_confirmation'] = data['volatility_momentum_convergence'] * data['liquidity_acceleration_factor']
    
    # Price-Volume Efficiency Integration
    # Movement Efficiency Components
    data['basic_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_efficiency'] = data['basic_efficiency'] * data['amount']
    data['efficiency_decay'] = data['volume_efficiency'] / data['volume_efficiency'].shift(1)
    
    # Volatility-Efficiency Interaction
    data['regime_adjusted_efficiency'] = data['basic_efficiency'] * data['volatility_compression_efficiency']
    data['efficiency_momentum_2'] = data['regime_adjusted_efficiency'] / data['regime_adjusted_efficiency'].shift(1)
    data['volatility_efficiency_divergence'] = data['efficiency_momentum_2'] * data['volatility_momentum']
    
    # Liquidity-Efficiency Signals
    data['volume_constrained_efficiency'] = data['basic_efficiency'] / data['volume_velocity']
    data['acceleration_efficiency'] = data['volume_constrained_efficiency'] * data['volume_acceleration']
    data['distribution_efficiency'] = data['acceleration_efficiency'] * data['distribution_skew']
    
    # Composite Alpha Factor Construction
    # Core Signal Generation
    data['volatility_efficiency_momentum'] = data['volatility_efficiency_divergence'] * data['efficiency_momentum_2']
    data['liquidity_constrained_signal'] = data['volatility_efficiency_momentum'] * data['liquidity_acceleration_factor']
    data['regime_adaptive_core'] = data['liquidity_constrained_signal'] * data['regime_transition_momentum']
    
    # Multi-Timeframe Validation
    data['short_term_confirmation'] = data['regime_adaptive_core'] * data['short_term_momentum_alignment']
    data['volatility_convergence_check'] = data['short_term_confirmation'] * data['volatility_convergence']
    data['liquidity_persistence'] = data['volatility_convergence_check'] * data['volume_persistence']
    
    # Final Alpha Factor Synthesis
    data['efficiency_weighted_signal'] = data['liquidity_persistence'] * data['efficiency_decay']
    data['distribution_adjusted_factor'] = data['efficiency_weighted_signal'] * data['distribution_skew']
    data['multi_timeframe_volatility_efficiency_alpha'] = data['distribution_adjusted_factor'] * data['volume_exhaustion']
    
    # Return the final alpha factor
    return data['multi_timeframe_volatility_efficiency_alpha']
