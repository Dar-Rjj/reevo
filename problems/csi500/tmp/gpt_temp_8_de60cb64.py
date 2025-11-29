import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Flow Geometric Analysis
    # Intraday range geometry
    data['range_geometry'] = (data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 1e-8)
    
    # Session volatility asymmetry (morning vs afternoon proxy)
    data['morning_range'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / 2
    data['afternoon_range'] = (data['high'] - data['low']) - data['morning_range']
    data['volatility_asymmetry'] = data['morning_range'] / (data['afternoon_range'] + 1e-8)
    
    # Volatility compression geometry
    data['volume_quantile'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volatility_compression'] = data['range_efficiency'] * data['volume_quantile']
    
    # Flow geometric distribution
    data['volume_flow_geometry'] = (data['amount'] / (data['volume'] + 1e-8)).rolling(window=10).std()
    data['price_flow_interaction'] = (data['close'] - data['open']) * data['volume_flow_geometry']
    
    # Flow fragmentation geometry
    data['volume_clustering'] = data['volume'].rolling(window=10).apply(
        lambda x: (x / x.mean()).std(), raw=False
    )
    data['price_clustering'] = (data['high'] - data['low']).rolling(window=10).std()
    data['flow_fragmentation'] = data['volume_clustering'] / (data['price_clustering'] + 1e-8)
    
    # Session Boundary Geometric Efficiency
    # Opening geometric momentum
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['first_hour_range'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min())
    data['gap_absorption'] = np.abs(data['opening_gap']) / (data['first_hour_range'] + 1e-8)
    
    # Opening range geometric efficiency
    data['opening_range_utilization'] = (data['close'].rolling(window=3).mean() - data['open']) / (
        data['first_hour_range'] + 1e-8
    )
    
    # Closing geometric flow patterns
    data['final_hour_decay'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / (
        data['high'] - data['low'] + 1e-8
    )
    
    # Final 15-minute vs 30-minute geometric divergence (proxy using last 2 vs last 4 periods)
    data['final_15min_move'] = data['close'] - data['close'].shift(2)
    data['final_30min_move'] = data['close'] - data['close'].shift(4)
    data['closing_asymmetry'] = data['final_15min_move'] - data['final_30min_move']
    
    # Session transition geometric efficiency
    data['morning_afternoon_consistency'] = (
        data['opening_range_utilization'].rolling(window=5).std() / 
        (data['final_hour_decay'].rolling(window=5).std() + 1e-8)
    )
    
    # Regime-Adaptive Geometric Alignment
    # Volatility regime characteristics
    data['volatility_regime'] = (data['high'] - data['low']).rolling(window=20).std()
    data['high_vol_geometry'] = data['range_geometry'] * (data['volatility_regime'] > data['volatility_regime'].quantile(0.7))
    data['low_vol_geometry'] = data['volatility_compression'] * (data['volatility_regime'] < data['volatility_regime'].quantile(0.3))
    
    # Regime transition geometric adaptation
    data['volatility_change'] = data['volatility_regime'].pct_change(5)
    data['regime_adaptation'] = data['range_efficiency'] / (np.abs(data['volatility_change']) + 1e-8)
    
    # Volume-price geometric alignment
    data['geometric_efficiency'] = np.abs(data['range_efficiency']) * data['volume_flow_geometry']
    data['session_consistency'] = (
        data['opening_range_utilization'].rolling(window=5).mean() * 
        data['final_hour_decay'].rolling(window=5).mean()
    )
    
    # Multi-scale geometric integration
    data['multi_scale_geometry'] = (
        data['range_geometry'].rolling(window=5).mean() * 
        data['volatility_compression'].rolling(window=10).mean()
    )
    
    # Final Composite Geometric Flow Factor
    # Multi-dimensional geometric scoring
    volatility_geometry_score = (
        data['range_geometry'].rolling(window=10).mean() * 
        data['flow_fragmentation'].rolling(window=10).mean()
    )
    
    session_boundary_score = (
        data['gap_absorption'].rolling(window=10).mean() * 
        data['closing_asymmetry'].rolling(window=10).mean()
    )
    
    regime_adaptive_score = (
        data['regime_adaptation'].rolling(window=10).mean() * 
        data['session_consistency'].rolling(window=10).mean()
    )
    
    # Geometric flow integration
    core_geometry = volatility_geometry_score * data['geometric_efficiency'].rolling(window=10).mean()
    session_transition = session_boundary_score * data['regime_adaptation'].rolling(window=10).mean()
    multi_scale_divergence = data['multi_scale_geometry'] * regime_adaptive_score
    
    # Composite geometric flow factor
    composite_factor = (
        core_geometry * 0.4 + 
        session_transition * 0.35 + 
        multi_scale_divergence * 0.25
    )
    
    # Final scoring with regime context
    volatility_context = data['volatility_regime'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    final_factor = composite_factor * (1 + volatility_context * 0.5)
    
    return final_factor
