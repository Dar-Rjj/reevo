import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic components
    data['close_prev'] = data['close'].shift(1)
    data['high_prev'] = data['high'].shift(1)
    data['low_prev'] = data['low'].shift(1)
    data['volume_prev2'] = data['volume'].shift(2)
    data['amount_prev2'] = data['amount'].shift(2)
    
    # Momentum Absorption Dynamics
    # Range-Adjusted Momentum Fade
    data['range_adjusted_momentum_fade'] = (data['close'] - data['close_prev']) / (data['high'] - data['low'])
    
    # Momentum Persistence Quality
    data['momentum_persistence_quality'] = (data['high'] - data['low']) / np.abs(data['close'] - data['close_prev'])
    data['momentum_persistence_quality'] = data['momentum_persistence_quality'].replace([np.inf, -np.inf], np.nan)
    
    # Momentum Breakdown Signal
    data['momentum_breakdown_signal'] = data['range_adjusted_momentum_fade'] * data['momentum_persistence_quality']
    
    # Liquidity Absorption Range Analysis
    data['range_volume_absorption'] = data['volume'] / (data['high'] - data['low'])
    data['range_volume_absorption'] = data['range_volume_absorption'].replace([np.inf, -np.inf], np.nan)
    
    data['range_volume_absorption_prev2'] = data['range_volume_absorption'].shift(2)
    data['absorption_momentum_divergence'] = data['range_volume_absorption'] - data['range_volume_absorption_prev2']
    
    data['liquidity_range_efficiency'] = data['absorption_momentum_divergence'] * data['momentum_breakdown_signal']
    
    # Range Efficiency with Momentum Integration
    # Wick-Based Momentum Efficiency
    data['upper_wick_momentum_pressure'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['lower_wick_momentum_support'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['net_momentum_efficiency'] = data['upper_wick_momentum_pressure'] - data['lower_wick_momentum_support']
    
    # Gap Momentum Dynamics
    data['opening_gap_momentum'] = (data['open'] - data['close_prev']) / data['close_prev']
    data['gap_fill_momentum_ratio'] = (data['high'] - data['low']) / np.abs(data['open'] - data['close_prev'])
    data['gap_fill_momentum_ratio'] = data['gap_fill_momentum_ratio'].replace([np.inf, -np.inf], np.nan)
    data['gap_momentum_efficiency'] = data['gap_fill_momentum_ratio'] * data['opening_gap_momentum']
    
    # Absorption-Efficiency Convergence
    # Momentum Absorption Quality
    data['liquidity_confirmed_breakdown'] = data['liquidity_range_efficiency'] * data['momentum_breakdown_signal']
    data['efficiency_enhanced_absorption'] = data['liquidity_confirmed_breakdown'] * data['net_momentum_efficiency']
    data['gap_aligned_absorption'] = data['efficiency_enhanced_absorption'] * data['gap_momentum_efficiency']
    
    # Range-Momentum Coherence
    data['daily_range_momentum_alignment'] = (data['high'] - data['low']) - (data['high_prev'] - data['low_prev'])
    data['range_position_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['range_momentum_convergence'] = data['daily_range_momentum_alignment'] * data['range_position_momentum']
    
    # Cross-Sectional Absorption Patterns
    # Amount-Driven Absorption Dynamics
    data['large_amount_absorption'] = data['amount'] / (data['high'] - data['low'])
    data['large_amount_absorption'] = data['large_amount_absorption'].replace([np.inf, -np.inf], np.nan)
    
    data['amount_absorption_momentum'] = (data['amount'] - data['amount_prev2']) / data['amount_prev2']
    data['amount_absorption_efficiency'] = data['large_amount_absorption'] * data['amount_absorption_momentum']
    
    # Opening Absorption Quality (using daily data as proxy)
    data['opening_absorption_ratio'] = data['volume'] / data['volume']  # Simplified as we don't have intraday data
    data['opening_range_efficiency'] = np.abs(data['open'] - data['close_prev']) / (data['high'] - data['low'])
    data['opening_absorption_signal'] = data['opening_absorption_ratio'] * data['opening_range_efficiency']
    
    # Alpha Factor Synthesis
    # Core Absorption-Efficiency Integration
    data['momentum_absorption_core'] = data['gap_aligned_absorption'] * data['range_momentum_convergence']
    data['efficiency_confirmed_absorption'] = data['momentum_absorption_core'] * data['amount_absorption_efficiency']
    
    # Cross-Sectional Validation
    data['opening_absorption_enhancement'] = data['efficiency_confirmed_absorption'] * data['opening_absorption_signal']
    
    data['opening_absorption_enhancement_prev1'] = data['opening_absorption_enhancement'].shift(1)
    data['opening_absorption_enhancement_prev3'] = data['opening_absorption_enhancement'].shift(3)
    data['multi_period_absorption_consistency'] = data['opening_absorption_enhancement_prev1'] * data['opening_absorption_enhancement_prev3']
    
    # Final Alpha Generation
    data['cross_sectional_absorption_score'] = data['multi_period_absorption_consistency']
    
    # Cross-sectional rank (within each day)
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    alpha_factor = data.groupby(data.index)['cross_sectional_absorption_score'].transform(cross_sectional_rank)
    
    return alpha_factor
