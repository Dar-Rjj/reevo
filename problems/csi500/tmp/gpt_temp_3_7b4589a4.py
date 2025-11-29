import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Fractal Dynamics Factor
    # Multi-Timeframe Volume Distribution Analysis
    data['morning_volume'] = data['volume'].rolling(window=5).apply(lambda x: x[:3].sum() / x.sum() if x.sum() > 0 else 0)
    data['afternoon_volume'] = data['volume'].rolling(window=5).apply(lambda x: x[3:].sum() / x.sum() if x.sum() > 0 else 0)
    data['volume_skewness'] = data['morning_volume'] - data['afternoon_volume']
    
    # Volume clustering persistence
    data['high_volume'] = data['volume'] > data['volume'].rolling(window=20).mean()
    data['consecutive_high_volume'] = data['high_volume'].rolling(window=5).sum()
    data['consecutive_low_volume'] = (~data['high_volume']).rolling(window=5).sum()
    
    # Volume regime transitions
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['volume_std'] = data['volume'].rolling(window=20).std()
    data['volume_breakout'] = (data['volume'] > data['volume_ma'] + data['volume_std']).astype(int)
    data['volume_collapse'] = (data['volume'] < data['volume_ma'] - data['volume_std']).astype(int)
    
    # Price response to volume patterns
    data['returns'] = data['close'].pct_change()
    data['high_volume_returns'] = data['returns'].where(data['high_volume'], 0)
    data['low_volume_returns'] = data['returns'].where(~data['high_volume'], 0)
    
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['volume_spike_range'] = data['price_range'].where(data['volume_breakout'] == 1, 0)
    data['volume_drop_range'] = data['price_range'].where(data['volume_collapse'] == 1, 0)
    
    # Fractal alignment
    data['volume_price_corr'] = data['volume'].rolling(window=10).corr(data['close'])
    data['intraday_alignment'] = data['morning_volume'].rolling(window=5).corr(data['returns'].rolling(window=5).mean())
    
    # Generate Fractal Momentum Signals
    data['volume_concentration_score'] = (data['consecutive_high_volume'] - data['consecutive_low_volume']) / 5
    data['price_response_score'] = data['high_volume_returns'].rolling(window=5).mean() - data['low_volume_returns'].rolling(window=5).mean()
    data['pattern_maturity'] = data['volume_price_corr'].abs().rolling(window=10).mean()
    
    fractal_factor = (data['volume_concentration_score'] * data['price_response_score'] * data['pattern_maturity']).fillna(0)
    
    # Range-Efficiency Wave Factor
    # Intraday Efficiency Cycles
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_ma'] = data['range_efficiency'].rolling(window=10).mean()
    data['efficiency_momentum'] = data['range_efficiency'] - data['efficiency_ma']
    
    # Volume-efficiency coupling
    data['efficiency_volume_corr'] = data['range_efficiency'].rolling(window=10).corr(data['volume'])
    
    # Efficiency regime boundaries
    data['efficiency_breakout'] = (data['range_efficiency'] > data['efficiency_ma'] + data['range_efficiency'].rolling(window=20).std()).astype(int)
    data['efficiency_breakdown'] = (data['range_efficiency'] < data['efficiency_ma'] - data['range_efficiency'].rolling(window=20).std()).astype(int)
    
    # Transition quality
    data['transition_cleanliness'] = abs(data['efficiency_momentum'].diff()).rolling(window=5).mean()
    data['volume_confirmation'] = data['volume_breakout'].where(data['efficiency_breakout'] == 1, 0) + data['volume_collapse'].where(data['efficiency_breakdown'] == 1, 0)
    
    # Generate Wave-Based Signals
    data['efficiency_returns'] = data['returns'].where(data['efficiency_momentum'] > 0, -data['returns'])
    data['transition_strength'] = data['transition_cleanliness'] * data['volume_confirmation']
    
    efficiency_factor = (data['efficiency_momentum'] * data['transition_strength'] * data['efficiency_volume_corr'].abs()).fillna(0)
    
    # Momentum-Flow Cascade Factor
    # Momentum propagation
    data['open_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['momentum_preservation'] = data['intraday_momentum'] / (data['open_momentum'].abs() + 1e-8)
    
    # Volume-momentum relationships
    data['momentum_volume_corr'] = data['intraday_momentum'].rolling(window=10).corr(data['volume'])
    data['volume_momentum_timing'] = data['volume'].shift(1).rolling(window=5).corr(data['intraday_momentum'])
    
    # Generate Cascade Signals
    data['momentum_flow_score'] = data['momentum_preservation'].clip(-2, 2)
    data['volume_alignment_score'] = (data['momentum_volume_corr'] + data['volume_momentum_timing']) / 2
    data['flow_maturity'] = data['momentum_volume_corr'].abs().rolling(window=10).mean()
    
    momentum_factor = (data['momentum_flow_score'] * data['volume_alignment_score'] * data['flow_maturity']).fillna(0)
    
    # Volatility-Clustering Regime Factor
    # Volatility concentration patterns
    data['volatility'] = data['returns'].abs()
    data['high_volatility'] = data['volatility'] > data['volatility'].rolling(window=20).mean()
    data['consecutive_high_vol'] = data['high_volatility'].rolling(window=5).sum()
    data['consecutive_low_vol'] = (~data['high_volatility']).rolling(window=5).sum()
    
    # Volatility regime transitions
    data['volatility_cluster_strength'] = data['consecutive_high_vol'] - data['consecutive_low_vol']
    
    # Price behavior in volatility regimes
    data['high_vol_returns'] = data['returns'].where(data['high_volatility'], 0)
    data['low_vol_returns'] = data['returns'].where(~data['high_volatility'], 0)
    
    # Generate Cluster-Based Signals
    data['clustering_intensity'] = data['volatility_cluster_strength'] / 5
    data['regime_return_relationship'] = data['high_vol_returns'].rolling(window=10).mean() - data['low_vol_returns'].rolling(window=10).mean()
    data['regime_maturity'] = data['volatility'].rolling(window=20).std() / data['volatility'].rolling(window=60).std()
    
    volatility_factor = (data['clustering_intensity'] * data['regime_return_relationship'] * data['regime_maturity']).fillna(0)
    
    # Liquidity-Wave Propagation Factor
    # Liquidity flow waves
    data['liquidity'] = data['volume'] * data['close']
    data['liquidity_ma'] = data['liquidity'].rolling(window=20).mean()
    data['liquidity_momentum'] = (data['liquidity'] - data['liquidity_ma']) / data['liquidity_ma']
    
    # Liquidity regime transitions
    data['liquidity_breakout'] = (data['liquidity'] > data['liquidity_ma'] + data['liquidity'].rolling(window=20).std()).astype(int)
    data['liquidity_breakdown'] = (data['liquidity'] < data['liquidity_ma'] - data['liquidity'].rolling(window=20).std()).astype(int)
    
    # Transition quality
    data['liquidity_transition_clean'] = abs(data['liquidity_momentum'].diff()).rolling(window=5).mean()
    data['liquidity_volume_confirmation'] = data['volume_breakout'].where(data['liquidity_breakout'] == 1, 0) + data['volume_collapse'].where(data['liquidity_breakdown'] == 1, 0)
    
    # Generate Wave Propagation Signals
    data['liquidity_returns'] = data['returns'].where(data['liquidity_momentum'] > 0, -data['returns'])
    data['liquidity_transition_strength'] = data['liquidity_transition_clean'] * data['liquidity_volume_confirmation']
    
    liquidity_factor = (data['liquidity_momentum'] * data['liquidity_transition_strength'] * data['liquidity_returns'].abs()).fillna(0)
    
    # Combine all factors with equal weights
    combined_factor = (
        fractal_factor.rank(pct=True) + 
        efficiency_factor.rank(pct=True) + 
        momentum_factor.rank(pct=True) + 
        volatility_factor.rank(pct=True) + 
        liquidity_factor.rank(pct=True)
    ) / 5
    
    return combined_factor
