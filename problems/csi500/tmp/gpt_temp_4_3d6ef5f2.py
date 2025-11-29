import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data.groupby(level=1)['close'].shift(1)
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Define midday as average of open and close (simplified approach)
    data['midday'] = (data['open'] + data['close']) / 2
    
    # Morning and afternoon fragmentation momentum
    data['morning_fragmentation_momentum'] = (data['midday'] - data['open']) / data['open']
    data['afternoon_fragmentation_momentum'] = (data['close'] - data['midday']) / data['midday']
    
    # Gap-momentum divergence
    data['gap_momentum_divergence'] = data['gap_magnitude'] - (data['morning_fragmentation_momentum'] + data['afternoon_fragmentation_momentum']) / 2
    
    # Range calculations
    data['daily_range'] = data['high'] - data['low']
    
    # Morning and afternoon range efficiency
    data['morning_range_efficiency'] = np.abs(data['midday'] - data['open']) / data['daily_range']
    data['afternoon_range_efficiency'] = np.abs(data['close'] - data['midday']) / data['daily_range']
    data['fragmentation_efficiency_divergence'] = data['morning_range_efficiency'] - data['afternoon_range_efficiency']
    
    # Volume calculations (assuming half day volumes - simplified)
    data['morning_volume'] = data['volume'] * 0.5
    data['afternoon_volume'] = data['volume'] * 0.5
    
    # Cross-sectional calculations
    def cross_sectional_avg(series):
        return series.groupby(level=0).transform('mean')
    
    # Gap context
    data['cross_sectional_gap_context'] = data['gap_magnitude'] - cross_sectional_avg(data['gap_magnitude'])
    
    # Morning and afternoon strength
    data['cross_sectional_morning_strength'] = data['morning_fragmentation_momentum'] - cross_sectional_avg(data['morning_fragmentation_momentum'])
    data['cross_sectional_afternoon_strength'] = data['afternoon_fragmentation_momentum'] - cross_sectional_avg(data['afternoon_fragmentation_momentum'])
    
    # Gap-fragmentation patterns
    data['gap_fragmentation_alignment'] = np.sign(data['gap_magnitude']) * np.sign(data['gap_momentum_divergence'])
    data['cross_sectional_gap_momentum_divergence'] = data['cross_sectional_gap_context'] * data['gap_fragmentation_alignment']
    data['gap_fragmentation_intensity'] = np.abs(data['gap_momentum_divergence']) * np.abs(data['fragmentation_efficiency_divergence'])
    
    # Volume-amount microstructure
    data['morning_volume_fragmentation'] = data['morning_volume'] * np.abs(data['morning_fragmentation_momentum'])
    data['afternoon_volume_release'] = data['afternoon_volume'] * np.abs(data['afternoon_fragmentation_momentum'])
    data['volume_fragmentation_ratio'] = data['morning_volume_fragmentation'] / (data['afternoon_volume_release'] + 1e-8)
    
    data['amount_per_unit_move'] = data['amount'] / (np.abs(data['close'] - data['open']) + 1e-8)
    data['cross_sectional_amount_efficiency'] = data['amount_per_unit_move'] - cross_sectional_avg(data['amount_per_unit_move'])
    
    # Volume context
    data['morning_volume_intensity'] = data['morning_volume'] / (data.groupby(level=1)['morning_volume'].transform(lambda x: x.rolling(5, min_periods=1).mean()) + 1e-8)
    data['afternoon_volume_persistence'] = data['afternoon_volume'] / (data['morning_volume'] + 1e-8)
    
    data['cross_sectional_morning_volume'] = data['morning_volume_intensity'] - cross_sectional_avg(data['morning_volume_intensity'])
    data['cross_sectional_afternoon_volume'] = data['afternoon_volume_persistence'] - cross_sectional_avg(data['afternoon_volume_persistence'])
    data['volume_momentum_correlation'] = data['cross_sectional_morning_volume'] * data['cross_sectional_morning_strength']
    
    # Microstructure confirmation
    data['volume_amount_fragmentation_sync'] = np.sign(data['volume_fragmentation_ratio']) * np.sign(data['cross_sectional_amount_efficiency'])
    data['microstructure_anchoring_strength'] = data['volume_fragmentation_ratio'] * data['cross_sectional_amount_efficiency']
    data['cross_sectional_volume_amount_regime'] = data['volume_momentum_correlation'] * data['microstructure_anchoring_strength']
    
    # Breakout detection
    data['prev_5d_high'] = data.groupby(level=1)['high'].transform(lambda x: x.rolling(5, min_periods=1).max().shift(1))
    
    data['morning_breakout_gap_fragmentation'] = ((data['midday'] - data['prev_5d_high']) / (data['prev_5d_high'] + 1e-8)) * data['gap_magnitude']
    data['afternoon_breakout_fragmentation'] = ((data['close'] - data['prev_5d_high']) / (data['prev_5d_high'] + 1e-8)) * data['afternoon_fragmentation_momentum']
    
    data['multi_session_breakout_fragmentation'] = (data['morning_breakout_gap_fragmentation'] + data['afternoon_breakout_fragmentation']) / 2
    data['cross_sectional_breakout_context'] = data['multi_session_breakout_fragmentation'] - cross_sectional_avg(data['multi_session_breakout_fragmentation'])
    
    # Volume-anchored regime confirmation
    data['breakout_volume_microstructure'] = data['volume'] * data['cross_sectional_breakout_context']
    data['amount_efficiency_in_breakout'] = data['cross_sectional_amount_efficiency'] * data['cross_sectional_breakout_context']
    data['volume_anchored_breakout'] = data['breakout_volume_microstructure'] * data['amount_efficiency_in_breakout']
    data['gap_breakout_alignment'] = np.sign(data['gap_magnitude']) * np.sign(data['cross_sectional_breakout_context'])
    
    # Integrated gap-breakout regime
    data['gap_breakout_fragmentation'] = data['gap_magnitude'] * data['cross_sectional_breakout_context']
    data['volume_confirmed_gap_breakout'] = data['gap_breakout_fragmentation'] * data['volume_anchored_breakout']
    data['regime_breakout_signal'] = data['volume_confirmed_gap_breakout'] * data['gap_breakout_alignment']
    
    # Regime transition analysis
    data['relative_absorption_rate'] = data['morning_range_efficiency'] / (np.abs(data['cross_sectional_gap_context']) + 1e-8)
    data['gap_absorption_momentum'] = data['relative_absorption_rate'] * data['cross_sectional_morning_strength']
    data['opening_pressure_core'] = data['gap_absorption_momentum'] * data['relative_absorption_rate']
    
    data['midday_momentum_continuation'] = np.sign(data['cross_sectional_morning_strength']) * np.sign(data['cross_sectional_afternoon_strength'])
    data['prev_5d_range_avg'] = data.groupby(level=1)['daily_range'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    data['relative_range_expansion'] = data['daily_range'] / (data['prev_5d_range_avg'] + 1e-8)
    data['volume_confirmed_continuation'] = data['cross_sectional_afternoon_volume'] * data['midday_momentum_continuation']
    data['midday_continuation_core'] = data['volume_confirmed_continuation'] * data['relative_range_expansion']
    
    data['relative_pressure_exhaustion'] = data['afternoon_range_efficiency'] * np.abs(data['cross_sectional_afternoon_strength'])
    data['volume_weighted_exhaustion'] = data['relative_pressure_exhaustion'] * data['cross_sectional_afternoon_volume']
    data['closing_exhaustion_core'] = data['volume_weighted_exhaustion'] * data['relative_pressure_exhaustion']
    
    # Performance context
    data['short_term_performance'] = data.groupby(level=1)['close'].pct_change(1)
    data['medium_term_context'] = data.groupby(level=1)['close'].pct_change(5)
    
    data['cross_sectional_short_term'] = data['short_term_performance'] - cross_sectional_avg(data['short_term_performance'])
    data['cross_sectional_medium_term'] = data['medium_term_context'] - cross_sectional_avg(data['medium_term_context'])
    data['short_medium_term_spread'] = data['cross_sectional_short_term'] - data['cross_sectional_medium_term']
    data['relative_acceleration'] = np.sign(data['short_medium_term_spread']) * data['cross_sectional_short_term']
    
    # Composite alpha construction
    data['gap_fragmentation_base'] = data['cross_sectional_gap_momentum_divergence'] * data['fragmentation_efficiency_divergence']
    data['microstructure_anchoring_core'] = data['gap_fragmentation_base'] * data['cross_sectional_volume_amount_regime']
    data['breakout_fragmentation_enhancement'] = data['microstructure_anchoring_core'] * data['regime_breakout_signal']
    
    data['opening_pressure_component'] = data['opening_pressure_core'] * data['cross_sectional_morning_strength']
    data['midday_continuation_component'] = data['midday_continuation_core'] * data['cross_sectional_afternoon_strength']
    data['closing_exhaustion_component'] = data['closing_exhaustion_core'] * np.abs(data['cross_sectional_afternoon_strength'])
    data['full_day_regime_signal'] = data['opening_pressure_component'] * data['midday_continuation_component'] * data['closing_exhaustion_component']
    
    data['gap_fragmentation_foundation'] = data['breakout_fragmentation_enhancement'] * data['gap_fragmentation_alignment']
    data['regime_context_enhancement'] = data['full_day_regime_signal'] * data['midday_momentum_continuation']
    data['performance_context'] = data['relative_acceleration'] * data['cross_sectional_gap_context']
    data['core_alpha_signal'] = data['gap_fragmentation_foundation'] * data['regime_context_enhancement'] * data['performance_context']
    
    data['volume_amount_confirmation'] = data['cross_sectional_volume_amount_regime'] * data['volume_amount_fragmentation_sync']
    data['regime_alignment'] = data['cross_sectional_gap_momentum_divergence'] * data['midday_momentum_continuation']
    data['validated_alpha_signal'] = data['core_alpha_signal'] * data['volume_amount_confirmation'] * data['regime_alignment']
    
    # Final alpha factor
    alpha_factor = data['validated_alpha_signal']
    
    return alpha_factor
