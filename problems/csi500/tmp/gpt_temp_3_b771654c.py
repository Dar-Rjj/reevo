import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic price components
    data['prev_close'] = data['close'].shift(1)
    data['prev_open'] = data['open'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Price Efficiency Component
    # Intraday Return Quality
    data['normalized_gap_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_capture_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_completion_score'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Momentum Persistence Analysis
    data['directional_consistency'] = np.sign(data['close'] - data['open']) * np.sign(data['open'] - data['prev_close'])
    
    # Calculate intraday return for momentum duration
    data['intraday_return'] = data['close'] - data['open']
    data['momentum_duration'] = 0
    for i in range(1, len(data)):
        if np.sign(data['intraday_return'].iloc[i]) == np.sign(data['intraday_return'].iloc[i-1]):
            data['momentum_duration'].iloc[i] = data['momentum_duration'].iloc[i-1] + 1
    
    data['persistence_strength'] = data['intraday_return'] * data['intraday_return'].rolling(window=3, min_periods=1).mean()
    
    # Price Rejection Patterns
    data['upper_shadow_dominance'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_shadow_dominance'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['shadow_asymmetry'] = data['upper_shadow_dominance'] - data['lower_shadow_dominance']
    
    # Volume Divergence System
    # Volume Pattern Analysis
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_trend_direction'] = data['volume'] / data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_stability'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).std()
    
    # Price-Volume Divergence
    data['return_volume_mismatch'] = np.abs(data['close'] - data['open']) * (1 / data['volume_compression'])
    data['directional_volume_alignment'] = np.sign(data['close'] - data['open']) * data['volume_trend_direction']
    data['volume_confirmation_score'] = data['volume_stability'] * np.abs(data['close'] - data['open'])
    
    # Volume Cluster Detection
    volume_median_10 = data['volume'].rolling(window=10, min_periods=1).median()
    data['volume_surge_indicator'] = (data['volume'] > 2 * volume_median_10).astype(int)
    data['volume_drought_indicator'] = (data['volume'] < 0.5 * volume_median_10).astype(int)
    data['volume_regime_score'] = data['volume_surge_indicator'] - data['volume_drought_indicator']
    
    # Range Expansion Dynamics
    # Range Development Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_expansion_ratio'] = data['daily_range'] / data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['range_position_efficiency'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / data['daily_range']
    data['range_asymmetry'] = (data['high'] - data['open']) / (data['open'] - data['low'])
    
    # Multi-timeframe Range Comparison
    data['short_term_range_momentum'] = data['daily_range'] / data['daily_range'].rolling(window=3, min_periods=1).mean()
    data['medium_term_range_baseline'] = data['daily_range'].rolling(window=10, min_periods=1).mean()
    data['range_acceleration'] = data['short_term_range_momentum'] / (data['daily_range'].rolling(window=5, min_periods=1).mean() / data['medium_term_range_baseline'])
    
    # Breakout Readiness Assessment
    data['compression_expansion_cycle'] = data['range_expansion_ratio'] * data['range_position_efficiency']
    data['range_boundary_proximity'] = np.minimum(data['high'] - data['close'], data['close'] - data['low']) / data['daily_range']
    data['expansion_quality'] = data['range_acceleration'] * data['range_asymmetry']
    
    # Volatility Regime Framework
    # Intraday Volatility Components
    data['upside_volatility'] = (data['high'] - data['open']) / data['open']
    data['downside_volatility'] = (data['open'] - data['low']) / data['open']
    data['volatility_skew'] = data['upside_volatility'] - data['downside_volatility']
    
    # Volatility Persistence
    data['volatility_momentum'] = data['daily_range'] / data['daily_range'].shift(1)
    data['volatility_regime_strength'] = data['daily_range'].rolling(window=5, min_periods=1).std() / data['daily_range'].rolling(window=10, min_periods=1).std()
    data['volatility_transition'] = np.abs(data['volatility_momentum'] - 1) * data['volatility_regime_strength']
    
    # Volatility-Volume Alignment
    data['high_volatility_confirmation'] = data['volatility_skew'] * data['volume_trend_direction']
    data['low_volatility_rejection'] = (1 / data['range_expansion_ratio']) * data['volume_compression']
    data['volatility_quality_score'] = data['volatility_transition'] * data['volume_confirmation_score']
    
    # Price-Volume Timing Component
    # Intraday Timing Signals
    data['early_session_momentum'] = (data['high'] - data['open']) / (data['open'] - data['low'])
    data['late_session_reversal'] = (data['close'] - (data['high'] + data['low'])/2) / data['daily_range']
    data['session_progression'] = (data['close'] - data['open']) / data['daily_range']
    
    # Multi-period Alignment
    data['prev_intraday_return'] = data['intraday_return'].shift(1)
    data['previous_day_carryover'] = np.sign(data['close'] - data['open']) * np.sign(data['prev_close'] - data['prev_open'])
    data['momentum_continuity'] = data['intraday_return'] * data['prev_intraday_return']
    
    # Calculate volatility regime persistence
    data['volatility_regime'] = (data['daily_range'] > data['daily_range'].rolling(window=10, min_periods=1).mean()).astype(int)
    data['regime_persistence'] = 0
    for i in range(1, len(data)):
        if data['volatility_regime'].iloc[i] == data['volatility_regime'].iloc[i-1]:
            data['regime_persistence'].iloc[i] = data['regime_persistence'].iloc[i-1] + 1
    
    # Timing Quality Assessment
    data['signal_clarity'] = np.abs(data['session_progression']) * data['volume_stability']
    data['timing_efficiency'] = data['early_session_momentum'] * data['late_session_reversal']
    data['execution_quality'] = data['signal_clarity'] * data['timing_efficiency']
    
    # Final Divergence Factor Integration
    # Core Divergence Signals
    data['price_volume_divergence'] = data['return_volume_mismatch'] * data['volume_regime_score']
    data['range_volatility_alignment'] = data['compression_expansion_cycle'] * data['volatility_quality_score']
    
    # Create momentum persistence analysis composite
    momentum_persistence_analysis = (
        data['directional_consistency'] + 
        data['persistence_strength'] / data['persistence_strength'].abs().max() +
        data['momentum_duration'] / data['momentum_duration'].max()
    )
    
    data['timing_momentum_integration'] = data['execution_quality'] * momentum_persistence_analysis
    
    # Signal Enhancement Filters
    enhanced_price_volume_divergence = data['price_volume_divergence'] * data['volume_regime_score']
    enhanced_range_volatility_alignment = data['range_volatility_alignment'] * data['range_boundary_proximity']
    enhanced_timing_momentum_integration = data['timing_momentum_integration'] * data['volatility_skew']
    
    # Multi-dimensional Integration
    # Calculate regime-weighted composite score
    regime_weight = 1 + (data['volatility_regime_strength'] * 0.5)
    
    # Apply persistence filtering
    persistence_filter = 1 + (data['regime_persistence'] / data['regime_persistence'].max() * 0.3)
    
    # Final Factor Calculation
    core_signals = (
        enhanced_price_volume_divergence + 
        enhanced_range_volatility_alignment + 
        enhanced_timing_momentum_integration
    )
    
    data['final_factor'] = (
        core_signals * 
        data['previous_day_carryover'] * 
        data['execution_quality'] * 
        regime_weight * 
        persistence_filter
    )
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
