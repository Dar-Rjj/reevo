import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Price-Volume Divergence Factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Volume-Price Efficiency Structure
    # Volume Concentration Analysis
    data['volume_high_ratio'] = data['volume'] * (data['high'] == data['close']).astype(int) / data['volume']
    data['volume_low_ratio'] = data['volume'] * (data['low'] == data['close']).astype(int) / data['volume']
    
    # Intraday Price-Volume Covariance
    data['price_range'] = data['high'] - data['low']
    data['price_volume_cov'] = (data['close'] - data['open']) * data['volume'] / data['price_range'].replace(0, np.nan)
    
    # Bidirectional Volume Pressure
    data['upward_pressure'] = np.where(data['close'] > data['open'], 
                                     (data['close'] - data['open']) * data['volume'], 0)
    data['downward_pressure'] = np.where(data['close'] < data['open'], 
                                       (data['open'] - data['close']) * data['volume'], 0)
    
    # High/Low Volume Concentration Ratios
    data['high_volume_concentration'] = data['volume_high_ratio'].rolling(window=5, min_periods=3).mean()
    data['low_volume_concentration'] = data['volume_low_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Price Path Efficiency Signals
    data['actual_distance'] = data['high'] - data['low']
    data['net_movement'] = abs(data['close'] - data['open'])
    data['path_efficiency'] = data['net_movement'] / data['actual_distance'].replace(0, np.nan)
    
    # Price Rejection Analysis
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['actual_distance'].replace(0, np.nan)
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['actual_distance'].replace(0, np.nan)
    data['body_to_range'] = abs(data['close'] - data['open']) / data['actual_distance'].replace(0, np.nan)
    
    # Volume-Weighted Price Positioning
    # Rolling volume-weighted calculations
    data['vw_high'] = (data['volume'] * data['high']).rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    data['vw_low'] = (data['volume'] * data['low']).rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    
    # Volume-Weighted Range Efficiency
    data['vw_range_efficiency'] = abs(data['close'] - data['open']) / (data['vw_high'] - data['vw_low']).replace(0, np.nan)
    
    # Price-Volume Divergence Metrics
    data['close_vw_position'] = (data['close'] - data['vw_low']) / (data['vw_high'] - data['vw_low']).replace(0, np.nan)
    data['range_volume_divergence'] = (data['actual_distance'] / data['price_range'].rolling(window=5, min_periods=3).mean()) - (data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean())
    
    # Multi-Session Divergence Patterns
    # Volume Pattern Persistence
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    data['volume_spike_cluster'] = (data['volume'] > data['volume'].rolling(window=10, min_periods=5).quantile(0.8)).rolling(window=3).sum()
    
    # Price-Volume Divergence Evolution
    data['divergence_magnitude'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['divergence_change'] = data['divergence_magnitude'].diff(3)
    
    # Cross-Session Efficiency Transfer
    data['prior_volume_impact'] = data['volume'].shift(1) / data['volume'].rolling(window=5, min_periods=3).mean()
    data['efficiency_momentum'] = data['path_efficiency'].rolling(window=3, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Enhanced Divergence Composite
    # Core Components
    data['volume_price_efficiency'] = (data['path_efficiency'] * data['price_volume_cov'] / data['price_volume_cov'].abs().rolling(window=10, min_periods=5).mean()).fillna(0)
    data['path_efficiency_momentum'] = data['path_efficiency'].pct_change(3).fillna(0)
    data['volume_concentration_bias'] = (data['high_volume_concentration'] - data['low_volume_concentration']).fillna(0)
    
    # Multi-dimensional Confirmation
    data['volume_dist_consistency'] = (data['volume_high_ratio'].rolling(window=3).std() + data['volume_low_ratio'].rolling(window=3).std()).fillna(0)
    data['price_path_validation'] = (data['upper_shadow'] + data['lower_shadow']).fillna(0)
    data['bidirectional_alignment'] = (data['upward_pressure'] - data['downward_pressure']) / (data['upward_pressure'] + data['downward_pressure']).replace(0, np.nan).fillna(0)
    
    # Multi-session Divergence Trend
    data['multi_session_trend'] = data['divergence_magnitude'].rolling(window=5, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True).fillna(0)
    
    # Composite Divergence Factor
    # Normalize components for cross-sectional comparison
    components = ['volume_price_efficiency', 'path_efficiency_momentum', 'volume_concentration_bias', 'bidirectional_alignment']
    
    for comp in components:
        if comp in data.columns:
            # Cross-sectional normalization (z-score within each day)
            data[f'{comp}_norm'] = data.groupby(data.index)[comp].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
    
    # Calculate final composite factor
    data['divergence_factor'] = (
        data.get('volume_price_efficiency_norm', 0) * 
        data.get('path_efficiency_momentum_norm', 0) * 
        data.get('volume_concentration_bias_norm', 0) * 
        data.get('bidirectional_alignment_norm', 0) * 
        (1 + data['multi_session_trend'])
    )
    
    # Final normalization
    result = data.groupby(data.index)['divergence_factor'].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
    
    return result
