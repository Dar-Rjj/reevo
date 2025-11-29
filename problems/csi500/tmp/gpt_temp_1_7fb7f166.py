import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close
    data['prev_close'] = data['close'].shift(1)
    
    # Intraday Pressure Components
    data['opening_pressure'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['high_pressure'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['low_pressure'] = (data['low'] - data['open']) / (data['open'] + 1e-8)
    
    # Demand Imbalance Detection
    data['daily_range'] = data['high'] - data['low']
    data['upward_demand'] = np.where(data['close'] > data['open'], 
                                   (data['close'] - data['low']) / (data['daily_range'] + 1e-8), 0)
    data['downward_pressure'] = np.where(data['close'] < data['open'], 
                                       (data['high'] - data['close']) / (data['daily_range'] + 1e-8), 0)
    data['demand_pressure_ratio'] = data['upward_demand'] / (data['downward_pressure'] + 0.001)
    
    # Volatility State Components
    data['daily_range_5d_mean'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['range_expansion_state'] = data['daily_range'] / (data['daily_range_5d_mean'] + 1e-8)
    data['volatility_persistence'] = data['daily_range'].rolling(window=3, min_periods=1).std()
    data['range_compression'] = 1 / (1 + data['range_expansion_state'])
    
    # Market Regime Classification
    data['high_vol_regime'] = (data['range_expansion_state'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['range_expansion_state'] < 0.8).astype(int)
    data['normal_regime'] = ((data['range_expansion_state'] >= 0.8) & 
                           (data['range_expansion_state'] <= 1.2)).astype(int)
    
    # Volume Pattern Components
    data['volume_3d_mean'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_intensity'] = data['volume'] / (data['volume_3d_mean'] + 1e-8)
    data['volume_concentration'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_5d_std'] = data['volume'].rolling(window=5, min_periods=1).std()
    data['volume_persistence'] = data['volume'] / (data['volume_5d_std'] + 1e-8)
    
    # Simplified Liquidity Flow Signals (using daily volume as proxy)
    data['early_session_liquidity'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['late_session_flow'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['liquidity_imbalance'] = data['early_session_liquidity'] - data['late_session_flow']
    
    # Multi-timeframe Pressure Integration
    data['high_pressure_2d'] = data['high_pressure'].rolling(window=2, min_periods=1).sum()
    data['low_pressure_2d'] = data['low_pressure'].rolling(window=2, min_periods=1).sum()
    data['pressure_differential'] = data['high_pressure_2d'] - data['low_pressure_2d']
    
    # Medium-term Regime Consistency
    data['vol_regime_5d'] = data['high_vol_regime'].rolling(window=5, min_periods=1).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
    data['vol_regime_10d'] = data['high_vol_regime'].rolling(window=10, min_periods=1).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
    data['regime_stability'] = (data['vol_regime_5d'] == data['vol_regime_10d']).astype(int)
    
    # Component Interaction and Synthesis
    data['core_pressure_signal'] = (data['opening_pressure'] * 
                                  data['pressure_differential'] * 
                                  data['demand_pressure_ratio'])
    
    data['volume_liquidity_signal'] = (data['volume_intensity'] * 
                                     data['liquidity_imbalance'] * 
                                     data['volume_concentration'])
    
    # Volatility Adaptation
    data['volatility_adaptation'] = data['core_pressure_signal'] * (1 + data['range_expansion_state'])
    data['volatility_adaptation'] = np.where(data['high_vol_regime'] == 1, 
                                           data['volatility_adaptation'], 
                                           data['core_pressure_signal'])
    
    # Regime-Adapted Signal
    data['regime_adapted_signal'] = data['volatility_adaptation'] * data['regime_stability']
    
    # Final Alpha Factor Construction
    data['combined_pressure_demand'] = data['regime_adapted_signal'] + data['volume_liquidity_signal']
    data['alpha_factor'] = data['combined_pressure_demand'].rolling(window=3, min_periods=1).mean()
    
    return data['alpha_factor']
