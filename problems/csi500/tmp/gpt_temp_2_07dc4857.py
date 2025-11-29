import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic components
    data['range'] = data['high'] - data['low']
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Intraday Mean Reversion Analysis
    data['intraday_deviation'] = (data['close'] - data['mid_price']) / data['range']
    data['range_expansion_ratio'] = data['range'] / data['range'].rolling(window=5).mean()
    data['reversion_potential'] = -np.sign(data['close'] - data['open']) * data['range_expansion_ratio']
    
    # Volatility Compression
    data['volatility_ratio'] = data['range'] / data['range'].rolling(window=10).mean()
    data['volatility_change'] = data['volatility_ratio'] - data['volatility_ratio'].shift(1)
    data['compression_signal'] = -data['volatility_change'] * data['range_expansion_ratio']
    
    # Price Level Context
    data['relative_position'] = (data['close'] - data['low']) / data['range']
    data['position_reversion'] = 0.5 - data['relative_position']
    data['level_adjusted_reversion'] = data['position_reversion'] * data['reversion_potential']
    
    # Volume Pattern Analysis
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['volume_range_ratio'] = data['volume'] / data['range']
    data['volume_efficiency'] = data['volume_range_ratio'] / data['volume_range_ratio'].rolling(window=5).mean()
    
    # Volatility-Volume Alignment
    data['vol_vol_convergence'] = data['volatility_ratio'] * data['volume_compression']
    data['convergence_change'] = data['vol_vol_convergence'] - data['vol_vol_convergence'].shift(1)
    data['convergence_momentum'] = data['convergence_change'] * data['volume_efficiency']
    
    # Generate Convergence Signal
    data['cumulative_convergence_momentum'] = data['convergence_momentum'].rolling(window=3).sum()
    data['convergence_range_alignment'] = data['cumulative_convergence_momentum'] * data['range_expansion_ratio']
    data['enhanced_convergence'] = data['convergence_range_alignment'] * data['volume_efficiency']
    
    # Pressure Distribution Analysis
    data['upper_pressure'] = (data['high'] - data['close']) * data['volume']
    data['lower_pressure'] = (data['close'] - data['low']) * data['volume']
    data['pressure_imbalance'] = (data['upper_pressure'] - data['lower_pressure']) / (data['upper_pressure'] + data['lower_pressure'])
    
    # Amount Flow Deceleration
    data['amount_change'] = data['amount'] / data['amount'].shift(1)
    data['amount_deceleration'] = data['amount_change'].shift(1) - data['amount_change']
    data['flow_slowdown'] = data['amount_deceleration'] * data['volume_compression']
    
    # Generate Dissipation Signal
    data['pressure_decay'] = data['pressure_imbalance'] * data['flow_slowdown']
    data['dissipation_momentum'] = data['pressure_decay'] - data['pressure_decay'].shift(1)
    data['enhanced_dissipation'] = data['dissipation_momentum'] * data['volume_efficiency']
    
    # Multi-Timeframe Divergence Detection
    data['short_term_reversion'] = (data['close'] / data['close'].shift(2) - 1) * data['reversion_potential']
    data['medium_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['reversion_momentum_divergence'] = data['short_term_reversion'] - data['medium_term_momentum']
    
    # Volume-Pressure Divergence
    data['volume_pressure_alignment'] = data['volume_compression'] * data['pressure_imbalance']
    data['divergence_signal'] = data['volume_pressure_alignment'] - data['reversion_momentum_divergence']
    data['enhanced_divergence'] = data['divergence_signal'] * data['flow_slowdown']
    
    # Volatility-Convergence Divergence
    data['vol_convergence_divergence'] = data['vol_vol_convergence'] - data['reversion_momentum_divergence']
    data['convergence_dissipation_alignment'] = data['convergence_momentum'] * data['dissipation_momentum']
    data['multi_scale_divergence'] = data['vol_convergence_divergence'] * data['convergence_dissipation_alignment']
    
    # Composite Factor Generation
    # Reversion Component
    reversion_component = data['level_adjusted_reversion'] * data['compression_signal'] * data['range_expansion_ratio']
    
    # Convergence Component
    convergence_component = data['enhanced_convergence'] * data['convergence_momentum'] * data['volume_efficiency']
    
    # Dissipation Component
    dissipation_component = data['enhanced_dissipation'] * data['flow_slowdown'] * data['pressure_decay']
    
    # Divergence Component
    divergence_component = data['multi_scale_divergence'] * data['enhanced_divergence'] * data['reversion_momentum_divergence']
    
    # Time-Decay Weighting
    weights = pd.Series([0.6, 0.4], index=[0, 1])
    weighted_reversion = reversion_component.rolling(window=2).apply(lambda x: (x * weights).sum(), raw=True)
    weighted_convergence = convergence_component.rolling(window=2).apply(lambda x: (x * weights).sum(), raw=True)
    weighted_dissipation = dissipation_component.rolling(window=2).apply(lambda x: (x * weights).sum(), raw=True)
    weighted_divergence = divergence_component.rolling(window=2).apply(lambda x: (x * weights).sum(), raw=True)
    
    # Final Composite Factor
    composite_factor = (weighted_reversion * weighted_convergence * 
                       weighted_dissipation * weighted_divergence)
    
    return composite_factor
