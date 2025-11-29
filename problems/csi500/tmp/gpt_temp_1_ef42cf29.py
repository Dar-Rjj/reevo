import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Cross-Sectional Price-Volume Asymmetry Factor
    # Calculate Directional Volume Pressure
    data['upward_move_high'] = np.where(data['high'] > data['open'], data['high'] - data['open'], 0)
    data['upward_move_close'] = np.where(data['close'] > data['open'], data['close'] - data['open'], 0)
    data['upside_pressure'] = (data['upward_move_high'] + data['upward_move_close']) * data['volume']
    
    data['downward_move_low'] = np.where(data['low'] < data['open'], data['open'] - data['low'], 0)
    data['downward_move_close'] = np.where(data['close'] < data['open'], data['open'] - data['close'], 0)
    data['downside_pressure'] = (data['downward_move_low'] + data['downward_move_close']) * data['volume']
    
    # Net Pressure Asymmetry
    data['net_pressure'] = data['upside_pressure'] - data['downside_pressure']
    data['total_pressure'] = data['upside_pressure'] + data['downside_pressure']
    data['pressure_imbalance'] = np.where(data['total_pressure'] > 0, data['net_pressure'] / data['total_pressure'], 0)
    data['pressure_persistence'] = data['pressure_imbalance'].rolling(window=3, min_periods=1).mean()
    
    # Analyze Intraday Price Response Efficiency
    data['price_change'] = data['close'] - data['open']
    data['price_elasticity_up'] = np.where((data['price_change'] > 0) & (data['volume'] > 0), 
                                         data['price_change'] / data['volume'], 0)
    data['price_elasticity_down'] = np.where((data['price_change'] < 0) & (data['volume'] > 0), 
                                           abs(data['price_change']) / data['volume'], 0)
    
    data['elasticity_ratio'] = np.where(data['price_elasticity_down'] > 0, 
                                      data['price_elasticity_up'] / data['price_elasticity_down'], 1)
    data['elasticity_vs_avg'] = data['elasticity_ratio'] / data['elasticity_ratio'].rolling(window=5, min_periods=1).mean()
    
    # Response Consistency
    data['response_stability'] = data['elasticity_ratio'].rolling(window=5, min_periods=1).std()
    data['response_degradation'] = data['elasticity_ratio'] / data['elasticity_ratio'].rolling(window=5, min_periods=1).mean() - 1
    
    # Generate Asymmetry-Based Alpha Signal
    data['pressure_accumulation'] = data['net_pressure'].rolling(window=2, min_periods=1).sum()
    data['response_pattern'] = data['elasticity_vs_avg'].rolling(window=8, min_periods=1).mean()
    
    # Combined signal logic
    data['asymmetry_signal'] = (
        data['pressure_persistence'] * 0.4 + 
        data['elasticity_vs_avg'] * 0.3 + 
        data['pressure_accumulation'] * 0.2 + 
        (1 - data['response_stability']) * 0.1
    )
    
    # Overnight Information Absorption Factor
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_size'] = abs(data['overnight_return'])
    data['recent_volatility'] = data['close'].pct_change().rolling(window=10, min_periods=1).std()
    data['gap_intensity'] = np.where(data['recent_volatility'] > 0, data['gap_size'] / data['recent_volatility'], 0)
    
    # Intraday Price Adjustment
    data['max_deviation'] = (data['high'] - data['low']) / data['open']
    data['adjustment_completion'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)
    data['adjustment_smoothness'] = 1 - (abs(data['close'] - (data['high'] + data['low']) / 2) / data['open'])
    
    # Volume Distribution Analysis
    data['opening_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).max()
    
    # Absorption Efficiency Signal
    data['absorption_efficiency'] = (
        data['gap_intensity'] * 0.3 + 
        data['adjustment_completion'] * 0.3 + 
        data['adjustment_smoothness'] * 0.2 + 
        data['opening_volume_ratio'] * 0.2
    )
    
    # Price-Volume Fractal Coherence Factor
    data['price_range'] = (data['high'] - data['low']) / data['open']
    data['body_size'] = abs(data['close'] - data['open']) / data['open']
    data['pattern_complexity'] = data['price_range'] * data['body_size'] * data['volume']
    
    # Volume Pattern Analysis
    data['volume_clustering'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_uniqueness'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).std()
    
    # Pattern Coherence
    data['price_volume_correlation'] = data['close'].pct_change().rolling(window=5, min_periods=1).corr(data['volume'].pct_change())
    data['coherence_divergence'] = (
        data['pattern_complexity'] * 0.4 + 
        data['volume_clustering'] * 0.3 + 
        abs(data['price_volume_correlation']) * 0.3
    )
    
    # Volatility-Clustered Momentum Persistence
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5, min_periods=1).std()
    data['volatility_cluster'] = data['volatility_5d'] / data['volatility_5d'].rolling(window=10, min_periods=1).mean()
    
    # Cluster-Adaptive Momentum
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_7d'] = data['close'].pct_change(7)
    
    # Adaptive strategy based on volatility cluster
    high_vol_mask = data['volatility_cluster'] > 1
    data['cluster_momentum'] = np.where(high_vol_mask, data['momentum_1d'], data['momentum_7d'])
    
    # Volume confirmation
    data['volume_momentum'] = data['volume'].pct_change(5)
    data['cluster_adaptive_score'] = data['cluster_momentum'] * (1 + data['volume_momentum'])
    
    # Micro-Structure Informed Momentum Factor
    data['price_autocorr'] = data['close'].pct_change().rolling(window=3, min_periods=1).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    
    data['volume_autocorr'] = data['volume'].pct_change().rolling(window=3, min_periods=1).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    
    # Micro-structure coordination
    data['micro_coordination'] = (
        abs(data['price_autocorr']) * 0.5 + 
        abs(data['volume_autocorr']) * 0.5
    )
    
    # Final combined factor
    data['final_factor'] = (
        data['asymmetry_signal'] * 0.25 +
        data['absorption_efficiency'] * 0.25 +
        data['coherence_divergence'] * 0.20 +
        data['cluster_adaptive_score'] * 0.15 +
        data['micro_coordination'] * 0.15
    )
    
    # Normalize the final factor
    factor_mean = data['final_factor'].rolling(window=20, min_periods=1).mean()
    factor_std = data['final_factor'].rolling(window=20, min_periods=1).std()
    data['normalized_factor'] = np.where(factor_std > 0, 
                                       (data['final_factor'] - factor_mean) / factor_std, 0)
    
    return data['normalized_factor']
