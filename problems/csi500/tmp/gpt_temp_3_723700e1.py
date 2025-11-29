import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Session Momentum with Microstructure Divergence factor
    """
    data = df.copy()
    
    # Basic calculations
    data['session_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['daily_range'] = data['high'] - data['low']
    data['daily_range_pct'] = data['daily_range'] / data['close'].shift(1)
    
    # Multi-Session Momentum Dynamics
    data['momentum_persistence'] = data['session_momentum'].rolling(window=3, min_periods=3).corr(data['session_momentum'].shift(1))
    data['momentum_reversal'] = data['session_momentum'] / data['session_momentum'].shift(1).replace(0, np.nan)
    
    # Volume-Weighted Momentum Quality
    data['upward_momentum'] = (data['session_momentum'] > 0).astype(int)
    data['momentum_volume_support'] = data['volume'] * data['upward_momentum']
    data['volume_change'] = data['volume'].pct_change()
    data['momentum_change'] = data['session_momentum'].pct_change()
    data['volume_momentum_consistency'] = data['volume_change'].rolling(window=5, min_periods=5).corr(data['momentum_change'])
    data['momentum_quality'] = data['session_momentum'] * data['volume_momentum_consistency']
    
    # Microstructure Flow Divergence
    # Using amount as proxy for transaction flow
    data['large_transaction_threshold'] = data['amount'].rolling(window=10, min_periods=10).quantile(0.8)
    data['large_transaction_dominance'] = data['amount'] / data['large_transaction_threshold'].replace(0, np.nan)
    
    # Session Flow Timing (simplified using volume patterns)
    data['flow_timing_divergence'] = (data['volume'].rolling(window=30, min_periods=30).apply(lambda x: x.iloc[:6].mean() if len(x) >= 6 else np.nan) / 
                                     data['volume'].rolling(window=30, min_periods=30).apply(lambda x: x.iloc[-6:].mean() if len(x) >= 6 else np.nan))
    
    # Range Expansion Quality
    data['range_expansion'] = data['daily_range'] / data['daily_range'].shift(1).replace(0, np.nan)
    data['range_persistence'] = data['range_expansion'].rolling(window=3, min_periods=3).corr(data['range_expansion'].shift(1))
    data['range_momentum'] = data['range_expansion'] / data['range_expansion'].rolling(window=3, min_periods=3).mean()
    
    # Volume-Confirmed Range Quality
    data['range_volume_support'] = data['volume'] * (data['range_expansion'] > 1).astype(int)
    data['volume_range_alignment'] = data['volume_change'].rolling(window=5, min_periods=5).corr(data['range_expansion'].pct_change())
    data['range_quality'] = data['range_expansion'] * data['volume_range_alignment']
    
    # Extreme Session Behavior
    data['session_extremeness'] = abs(data['close'] - data['open']) / data['daily_range'].replace(0, np.nan)
    data['volume_extremeness'] = data['volume'] / data['volume'].rolling(window=10, min_periods=10).mean()
    data['extreme_session_score'] = data['session_extremeness'] * data['volume_extremeness']
    
    # Cross-Dimensional Microstructure Integration
    data['momentum_flow_alignment'] = data['session_momentum'] * data['large_transaction_dominance']
    data['range_flow_confirmation'] = data['range_expansion'] * data['large_transaction_dominance']
    data['extreme_flow_dynamics'] = data['extreme_session_score'] * data['large_transaction_dominance']
    
    # Microstructure Divergence Detection
    data['momentum_flow_divergence'] = data['session_momentum'] - data['large_transaction_dominance'].rolling(window=5, min_periods=5).mean()
    data['range_flow_inconsistency'] = data['range_expansion'] - data['large_transaction_dominance'].rolling(window=5, min_periods=5).mean()
    
    # Final Alpha Construction
    # Positive signals
    data['aligned_momentum_signal'] = ((data['momentum_flow_alignment'] > data['momentum_flow_alignment'].rolling(window=10, min_periods=10).mean()) & 
                                      (data['momentum_quality'] > 0)).astype(int)
    
    data['confirmed_range_signal'] = ((data['range_flow_confirmation'] > data['range_flow_confirmation'].rolling(window=10, min_periods=10).mean()) & 
                                     (data['range_quality'] > 0)).astype(int)
    
    data['consistent_extreme_signal'] = ((data['extreme_flow_dynamics'] > data['extreme_flow_dynamics'].rolling(window=10, min_periods=10).mean()) & 
                                        (data['extreme_session_score'] > data['extreme_session_score'].rolling(window=10, min_periods=10).mean())).astype(int)
    
    # Negative signals
    data['diverging_momentum_signal'] = ((data['momentum_flow_divergence'] > data['momentum_flow_divergence'].rolling(window=10, min_periods=10).std()) & 
                                        (data['momentum_quality'] < 0)).astype(int)
    
    data['unsupported_range_signal'] = ((data['range_flow_inconsistency'] > data['range_flow_inconsistency'].rolling(window=10, min_periods=10).std()) & 
                                       (data['range_quality'] < 0)).astype(int)
    
    # Cross-Sectional Microstructure Score
    positive_signals = data['aligned_momentum_signal'] + data['confirmed_range_signal'] + data['consistent_extreme_signal']
    negative_signals = data['diverging_momentum_signal'] + data['unsupported_range_signal']
    
    data['microstructure_quality'] = (data['momentum_quality'].fillna(0) + 
                                     data['range_quality'].fillna(0) + 
                                     data['extreme_session_score'].fillna(0)) / 3
    
    # Final alpha factor
    data['alpha_factor'] = (positive_signals - negative_signals) * data['microstructure_quality']
    
    # Normalize and return
    alpha_series = data['alpha_factor'].fillna(0)
    return alpha_series
