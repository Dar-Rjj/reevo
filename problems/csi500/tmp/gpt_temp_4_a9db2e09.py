import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Volatility Momentum Convergence Factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency-Volatility Patterns
    data['fractal_directional_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['opening_momentum_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    data['intraday_volatility_efficiency'] = (data['high'] - data['low']) / (data['high'].rolling(window=5).mean() - data['low'].rolling(window=5).mean() + 1e-8)
    data['volatility_momentum_asymmetry'] = data['opening_momentum_efficiency'] * data['intraday_volatility_efficiency']
    
    # Session Efficiency Asymmetry (using open-high vs close-low as proxy)
    data['session_efficiency_asymmetry'] = ((data['high'] - data['open']) - (data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-Pressure Dynamics Synthesis
    data['volume_price_efficiency_ratio'] = data['close'] * data['volume'] / (data['amount'] + 1e-8)
    
    # Bullish/Bearish Pressure Intensity (using price movement direction)
    data['bullish_pressure'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['bearish_pressure'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['net_pressure_momentum'] = (data['bullish_pressure'].rolling(window=3).mean() - data['bearish_pressure'].rolling(window=3).mean()) * (data['volume'] / data['volume'].shift(1))
    
    # Volume Pattern Asymmetry (using first/last hour proxy - assuming first/last 1/6 of trading hours)
    data['volume_pattern_asymmetry'] = (data['volume'].rolling(window=2).apply(lambda x: x.iloc[0] if len(x) == 2 else np.nan) / 
                                      data['volume'].rolling(window=2).apply(lambda x: x.iloc[-1] if len(x) == 2 else np.nan)) * \
                                     ((data['volume'].rolling(window=2).apply(lambda x: x.iloc[0] if len(x) == 2 else np.nan) - 
                                       data['volume'].rolling(window=2).apply(lambda x: x.iloc[-1] if len(x) == 2 else np.nan)) / 
                                      (data['volume'].rolling(window=2).apply(lambda x: x.iloc[0] if len(x) == 2 else np.nan + 
                                                                             data['volume'].rolling(window=2).apply(lambda x: x.iloc[-1] if len(x) == 2 else np.nan) + 1e-8)))
    
    # Efficiency-Volatility Breakout Detection
    data['previous_high'] = data['high'].shift(1)
    data['fractal_breakout_strength'] = (data['close'] - data['previous_high']) / (data['previous_high'] + 1e-8)
    
    # Volume-Momentum Confirmed Breakout
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_momentum_confirmed_breakout'] = data['fractal_breakout_strength'] * (data['volume'] / (data['volume_20d_avg'] + 1e-8))
    
    # Efficiency-Volatility Weighted Breakout
    data['efficiency_volatility_weighted_breakout'] = data['fractal_breakout_strength'] * data['volatility_momentum_asymmetry']
    
    # Cross-Sectional Momentum Structure Analysis
    data['path_deviation'] = abs(data['fractal_directional_efficiency'] - ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)))
    data['volume_price_alignment'] = np.sign(data['close'] - data['open']) * (data['volume'] / (data['volume'].shift(1) + 1e-8))
    
    # True Range calculation
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['volatility_adjusted_efficiency'] = (data['close'] - data['open']) / (data['true_range'] + 1e-8)
    
    # Multi-period Efficiency-Volatility Analysis
    data['cumulative_efficiency_volatility_flow'] = (data['volatility_momentum_asymmetry'] * data['volume_pattern_asymmetry']).rolling(window=5).sum()
    data['volume_pressure_convergence'] = data['cumulative_efficiency_volatility_flow'] * data['net_pressure_momentum']
    data['volatility_adjusted_convergence'] = data['volume_pressure_convergence'] / (data['high'] - data['low'] + 1e-8)
    
    # Session Structure Integration (using open-high and close-low as proxy for morning/afternoon)
    data['morning_afternoon_volatility_ratio'] = (data['high'] - data['open']) / (data['close'] - data['low'] + 1e-8)
    data['volume_efficiency_alignment'] = data['volume_pattern_asymmetry'] * data['volatility_momentum_asymmetry']
    data['reversal_timing_efficiency'] = data['fractal_directional_efficiency'] * data['session_efficiency_asymmetry']
    
    # Composite Efficiency-Volatility Alpha Generation
    # Efficiency-Volatility Convergence Core
    data['directional_volatility_efficiency'] = data['volatility_momentum_asymmetry'] * data['volume_momentum_confirmed_breakout']
    data['pressure_enhanced_convergence'] = data['directional_volatility_efficiency'] * data['net_pressure_momentum']
    data['multi_period_convergence_factor'] = data['volume_pressure_convergence'] * data['volatility_adjusted_convergence']
    
    # Cross-Sectional Momentum Integration
    data['path_efficiency_component'] = data['fractal_directional_efficiency'] * data['path_deviation']
    data['volume_structure_alignment'] = data['volume_price_alignment'] * data['volume_pattern_asymmetry']
    data['cross_sectional_momentum_factor'] = data['path_efficiency_component'] * data['volume_structure_alignment']
    
    # Efficiency-Volatility Breakout Indicator
    data['breakout_efficiency_component'] = data['fractal_breakout_strength'] * data['volatility_momentum_asymmetry']
    data['reversal_confirmation'] = data['breakout_efficiency_component'] * data['volume_pattern_asymmetry']
    data['pressure_weighted_breakout_reversal'] = data['reversal_confirmation'] * data['volume_efficiency_alignment']
    
    # Composite Efficiency-Volatility Alpha
    data['core_convergence_score'] = data['directional_volatility_efficiency'] * data['pressure_enhanced_convergence']
    data['cross_sectional_enhancement'] = data['core_convergence_score'] * data['cross_sectional_momentum_factor']
    data['timing_enhancement'] = data['cross_sectional_enhancement'] * data['reversal_timing_efficiency']
    data['final_alpha_factor'] = data['timing_enhancement'] * data['multi_period_convergence_factor']
    
    # Return the final alpha factor series
    return data['final_alpha_factor']
