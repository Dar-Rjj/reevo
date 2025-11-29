import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Adjusted Breakout Momentum factor combining multiple dimensions:
    - Multi-Timeframe Breakout Efficiency
    - Gap Persistence with Efficiency Scoring  
    - Volatility-Regime Momentum Divergence
    - Amount-Weighted Efficiency Clustering
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Timeframe Breakout Efficiency
    # Calculate historical high-low ranges
    data['daily_range'] = data['high'] - data['low']
    data['range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Breakout strength calculation
    data['breakout_ratio'] = data['daily_range'] / data['range_20d']
    
    # Efficiency ratio calculation
    data['directional_move'] = abs(data['close'] - data['open'])
    data['efficiency_ratio'] = data['directional_move'] / (data['daily_range'] + 1e-8)
    
    # Volume analysis for confirmation
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    data['volume_trend'] = data['volume'].rolling(window=3, min_periods=2).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[-2] else -1 if x.iloc[-1] < x.iloc[-2] else 0
    )
    
    # Efficiency-adjusted breakout
    data['efficiency_breakout'] = data['breakout_ratio'] * data['efficiency_ratio']
    data['volume_confirmation'] = np.where(
        (data['efficiency_ratio'] > 0.6) & (data['volume_ratio'] > 1.2), 1.0,
        np.where((data['efficiency_ratio'] < 0.3) & (data['volume_ratio'] < 0.8), -1.0, 0.0)
    )
    
    # 2. Gap Persistence with Efficiency Scoring
    data['prev_close'] = data['close'].shift(1)
    data['gap_size'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['gap_abs'] = abs(data['gap_size'])
    
    # Gap efficiency assessment
    data['prev_range'] = data['daily_range'].shift(1)
    data['gap_relative'] = data['gap_abs'] / (data['prev_range'] / data['prev_close'] + 1e-8)
    
    # Gap persistence tracking
    data['gap_filled'] = np.where(
        (data['gap_size'] > 0) & (data['low'] <= data['prev_close']), 1,
        np.where((data['gap_size'] < 0) & (data['high'] >= data['prev_close']), 1, 0)
    )
    data['gap_remaining'] = np.where(
        data['gap_filled'] == 0,
        abs(data['close'] - data['prev_close']) / abs(data['open'] - data['prev_close'] + 1e-8),
        0
    )
    
    # Gap efficiency score
    data['gap_efficiency'] = data['efficiency_ratio'] * (1 - data['gap_filled'])
    data['gap_signal'] = np.sign(data['gap_size']) * data['gap_remaining'] * data['gap_efficiency']
    
    # 3. Volatility-Regime Momentum Divergence
    # Volatility analysis
    data['vol_ratio'] = data['range_5d'] / (data['range_20d'] + 1e-8)
    
    # Momentum calculations
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)
    data['momentum_20d'] = (data['close'] - data['close'].shift(20)) / (data['close'].shift(20) + 1e-8)
    
    # Efficiency-adjusted momentum
    data['eff_momentum_5d'] = data['momentum_5d'] * data['efficiency_ratio'].rolling(window=5, min_periods=3).mean()
    data['eff_momentum_20d'] = data['momentum_20d'] * data['efficiency_ratio'].rolling(window=20, min_periods=10).mean()
    
    # Momentum divergence
    data['momentum_divergence'] = data['eff_momentum_5d'] - data['eff_momentum_20d']
    
    # Volatility regime adjustments
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 'high',
                         np.where(data['vol_ratio'] < 0.8, 'low', 'normal'))
    
    data['regime_adjusted_div'] = data['momentum_divergence'] * np.where(
        data['vol_regime'] == 'high', 0.7,
        np.where(data['vol_regime'] == 'low', 1.3, 1.0)
    )
    
    # 4. Amount-Weighted Efficiency Clustering
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_ratio'] = data['amount'] / (data['amount_5d_avg'] + 1e-8)
    
    # Efficiency clustering
    data['high_efficiency'] = (data['efficiency_ratio'] > 0.6).astype(int)
    data['efficiency_cluster'] = data['high_efficiency'].rolling(window=3, min_periods=2).sum()
    
    # Amount-weighted efficiency
    data['amount_efficiency'] = data['efficiency_ratio'] * data['amount_ratio']
    data['cluster_strength'] = data['efficiency_cluster'] * data['amount_efficiency']
    
    # Cluster momentum
    data['cluster_momentum'] = data['cluster_strength'].diff(3)
    
    # 5. Cross-Sectional Factor Integration
    # Normalize individual components
    components = ['efficiency_breakout', 'gap_signal', 'regime_adjusted_div', 'cluster_momentum']
    
    for component in components:
        if component in data.columns:
            # Z-score normalization within each day (cross-sectional)
            data[f'{component}_norm'] = data.groupby(data.index)[component].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    # Multi-dimensional scoring with weights
    weights = {
        'efficiency_breakout_norm': 0.3,
        'gap_signal_norm': 0.25, 
        'regime_adjusted_div_norm': 0.25,
        'cluster_momentum_norm': 0.2
    }
    
    # Calculate final factor score
    factor_score = pd.Series(0.0, index=data.index)
    for component, weight in weights.items():
        if component in data.columns:
            factor_score += data[component] * weight
    
    # Volume confirmation multiplier
    volume_multiplier = np.where(
        data['volume_confirmation'] != 0, 
        1.2,  # Boost signals with volume confirmation
        0.8   # Reduce signals without volume confirmation
    )
    
    final_factor = factor_score * volume_multiplier
    
    return final_factor
