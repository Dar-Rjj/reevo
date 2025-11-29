import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Session Fractal Momentum Framework
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Create working copy
    data = df.copy()
    
    # Calculate first hour data (assuming first hour is first 1/6.5 of trading day)
    # For simplicity, we'll use the first hour as the first period
    data['prev_close'] = data['close'].shift(1)
    
    # Morning session components (first hour)
    data['morning_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.max())
    data['morning_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.min())
    data['first_hour_close'] = data['close'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.mean())
    
    # Morning Price Efficiency
    data['morning_price_efficiency'] = np.abs(data['first_hour_close'] - data['open']) / (data['morning_high'] - data['morning_low'] + 1e-8)
    
    # Morning Volume Efficiency
    data['morning_volume_efficiency'] = data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.mean()) / (data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.mean()) * (data['morning_high'] - data['morning_low'] + 1e-8) + 1e-8)
    
    # Morning Range Efficiency
    data['morning_range_efficiency'] = data['morning_price_efficiency'] * data['morning_volume_efficiency']
    
    # Afternoon session components
    # Afternoon Price Efficiency
    data['afternoon_price_efficiency'] = np.abs(data['close'] - data['first_hour_close']) / (data['high'] - data['low'] + 1e-8)
    
    # Afternoon Volume Efficiency
    data['afternoon_volume_efficiency'] = (data['amount'] - data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else 0)) / ((data['volume'] - data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else 0)) * (data['high'] - data['low'] + 1e-8) + 1e-8)
    
    # Afternoon Range Efficiency
    data['afternoon_range_efficiency'] = data['afternoon_price_efficiency'] * data['afternoon_volume_efficiency']
    
    # Cross-Session Efficiency Integration
    data['session_efficiency_ratio'] = data['morning_range_efficiency'] / (data['afternoon_range_efficiency'] + 1e-8)
    data['session_efficiency_alignment'] = np.sign(data['morning_range_efficiency']) * np.sign(data['afternoon_range_efficiency'])
    data['combined_session_efficiency'] = data['session_efficiency_ratio'] * data['session_efficiency_alignment']
    
    # Session Momentum Dynamics
    # Morning Micro Momentum
    data['morning_micro_momentum'] = (data['first_hour_close'] - data['open']) * data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.mean())
    
    # Afternoon Midpoint Momentum
    data['afternoon_midpoint_momentum'] = (data['close'] - data['first_hour_close']) * data['amount']
    
    # Cross-Session Medium Momentum
    data['cross_session_medium_momentum'] = (data['close'] - data['prev_close']) * (data['high'] - data['low'])
    
    # Session Momentum Regimes
    data['momentum_sign_product'] = np.sign(data['morning_micro_momentum']) * np.sign(data['afternoon_midpoint_momentum']) * np.sign(data['cross_session_medium_momentum'])
    data['convergent_momentum'] = data['momentum_sign_product'] > 0
    data['divergent_momentum'] = data['momentum_sign_product'] < 0
    data['neutral_momentum'] = ~(data['convergent_momentum'] | data['divergent_momentum'])
    
    # Momentum-Efficiency Integration
    data['morning_momentum_efficiency'] = data['morning_micro_momentum'] * data['morning_range_efficiency']
    data['afternoon_momentum_efficiency'] = data['afternoon_midpoint_momentum'] * data['afternoon_range_efficiency']
    data['cross_session_momentum_efficiency'] = data['cross_session_medium_momentum'] * data['combined_session_efficiency']
    
    # Session Gap Analysis
    # Morning Gap Structure
    data['overnight_gap'] = (data['open'] - data['prev_close']) * data['volume']
    data['morning_gap_resolution'] = data['overnight_gap'] * data['morning_micro_momentum']
    data['gap_efficiency_signal'] = data['morning_gap_resolution'] * data['morning_volume_efficiency']
    
    # Afternoon Gap Dynamics
    data['afternoon_gap_impact'] = (data['close'] - data['first_hour_close']) * data['amount']
    data['cross_session_gap_alignment'] = np.sign(data['overnight_gap']) * np.sign(data['afternoon_gap_impact'])
    data['gap_persistence'] = data['afternoon_gap_impact'] * data['cross_session_gap_alignment']
    
    # Session Gap Integration
    data['morning_gap_momentum'] = data['gap_efficiency_signal'] * data['morning_momentum_efficiency']
    data['afternoon_gap_momentum'] = data['gap_persistence'] * data['afternoon_momentum_efficiency']
    data['combined_gap_core'] = data['morning_gap_momentum'] + data['afternoon_gap_momentum']
    
    # Session Volume-Efficiency Patterns
    # Morning Volume Distribution
    data['morning_volume_concentration'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x.mean()) / (data['volume'] + 1e-8)
    data['morning_volume_efficiency_combined'] = data['morning_volume_efficiency'] * data['morning_volume_concentration']
    data['morning_volume_signal'] = data['morning_volume_efficiency_combined'] * data['morning_price_efficiency']
    
    # Afternoon Volume Distribution
    data['afternoon_volume_concentration'] = (data['volume'] - data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else 0)) / (data['volume'] + 1e-8)
    data['afternoon_volume_efficiency_combined'] = data['afternoon_volume_efficiency'] * data['afternoon_volume_concentration']
    data['afternoon_volume_signal'] = data['afternoon_volume_efficiency_combined'] * data['afternoon_price_efficiency']
    
    # Cross-Session Volume Integration
    data['session_volume_pattern'] = data['morning_volume_signal'] * data['afternoon_volume_signal']
    data['volume_efficiency_alignment'] = np.sign(data['morning_volume_signal']) * np.sign(data['afternoon_volume_signal'])
    data['combined_volume_efficiency'] = data['session_volume_pattern'] * data['volume_efficiency_alignment']
    
    # Session Fractal Regime Classification
    # Efficiency-Momentum Regimes
    data['high_efficiency_momentum'] = (data['combined_session_efficiency'] > 1.2) & data['convergent_momentum']
    data['low_efficiency_momentum'] = (data['combined_session_efficiency'] < 0.8) & data['divergent_momentum']
    data['transition_regime'] = ~(data['high_efficiency_momentum'] | data['low_efficiency_momentum'])
    
    # Volume-Efficiency Regimes
    data['concentrated_session'] = (data['morning_volume_concentration'] > 0.25) & (data['afternoon_volume_concentration'] < 0.6)
    data['distributed_session'] = (data['morning_volume_concentration'] < 0.15) & (data['afternoon_volume_concentration'] > 0.8)
    data['balanced_session'] = ~(data['concentrated_session'] | data['distributed_session'])
    
    # Gap-Momentum Regimes
    gap_core_abs = np.abs(data['combined_gap_core'])
    gap_70th = gap_core_abs.rolling(window=20, min_periods=10).quantile(0.7)
    gap_30th = gap_core_abs.rolling(window=20, min_periods=10).quantile(0.3)
    
    data['strong_gap_momentum'] = gap_core_abs > gap_70th
    data['moderate_gap_momentum'] = (gap_core_abs >= gap_30th) & (gap_core_abs <= gap_70th)
    data['weak_gap_momentum'] = gap_core_abs < gap_30th
    
    # Intraday Session Fractal Alpha Construction
    # Base Session Components
    data['efficiency_momentum_base'] = data['cross_session_momentum_efficiency'] * data['combined_session_efficiency']
    data['volume_efficiency_base'] = data['combined_volume_efficiency'] * data['session_volume_pattern']
    data['gap_momentum_base'] = data['combined_gap_core'] * data['cross_session_gap_alignment']
    
    # Regime-Adaptive Scaling
    data['regime_adaptive_alpha'] = 0.0
    # High Efficiency-Momentum Alpha
    high_mask = data['high_efficiency_momentum']
    data.loc[high_mask, 'regime_adaptive_alpha'] = (data['efficiency_momentum_base'] * data['volume_efficiency_base'] * data['gap_momentum_base'])
    
    # Low Efficiency-Momentum Alpha
    low_mask = data['low_efficiency_momentum']
    data.loc[low_mask, 'regime_adaptive_alpha'] = ((data['efficiency_momentum_base'] + data['volume_efficiency_base']) * data['gap_momentum_base'])
    
    # Transition Regime Alpha
    trans_mask = data['transition_regime']
    data.loc[trans_mask, 'regime_adaptive_alpha'] = (data['efficiency_momentum_base'] * data['volume_efficiency_base'])
    
    # Final Session Fractal Alpha
    data['final_alpha'] = data['regime_adaptive_alpha'].copy()
    
    # Concentrated Session Enhancement
    conc_mask = data['concentrated_session']
    data.loc[conc_mask, 'final_alpha'] = data['regime_adaptive_alpha'] * data['morning_volume_signal']
    
    # Distributed Session Enhancement
    dist_mask = data['distributed_session']
    data.loc[dist_mask, 'final_alpha'] = data['regime_adaptive_alpha'] * data['afternoon_volume_signal']
    
    # Balanced Session Enhancement
    bal_mask = data['balanced_session']
    data.loc[bal_mask, 'final_alpha'] = data['regime_adaptive_alpha'] * data['combined_volume_efficiency']
    
    # Return the final alpha factor
    return data['final_alpha']
