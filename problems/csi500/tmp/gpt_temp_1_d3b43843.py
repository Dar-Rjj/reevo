import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Volatility Regime Adaptive Price-Volume Momentum factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Normalized Volatility
    data['norm_vol'] = data['true_range'] / data['prev_close']
    
    # Volatility Ratio
    data['vol_ratio'] = data['norm_vol'] / data['norm_vol'].rolling(window=10, min_periods=5).mean()
    
    # Regime Classification
    conditions = [
        data['vol_ratio'] > 1.5,
        (data['vol_ratio'] >= 0.8) & (data['vol_ratio'] <= 1.5),
        data['vol_ratio'] < 0.8
    ]
    choices = [2, 1, 0]  # 2=High, 1=Normal, 0=Low
    data['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Volatility Persistence Analysis
    data['regime_duration'] = 1
    for i in range(1, len(data)):
        if data['vol_regime'].iloc[i] == data['vol_regime'].iloc[i-1]:
            data.loc[data.index[i], 'regime_duration'] = data['regime_duration'].iloc[i-1] + 1
    
    # Volatility Momentum
    data['vol_momentum'] = data['vol_ratio'] - data['vol_ratio'].shift(1)
    
    # Adaptive Price Momentum Framework
    # Regime-Specific Momentum Calculations
    data['high_vol_momentum'] = (data['close'] - data['open']) / data['true_range']
    data['normal_vol_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['low_vol_momentum'] = (data['close'] - data['close'].rolling(window=10, min_periods=5).mean()) / data['close'].rolling(window=10, min_periods=5).mean()
    
    # Momentum Quality Assessment
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Directional Consistency
    data['ret_1d'] = data['close'].pct_change()
    data['direction_consistency'] = data['ret_1d'].rolling(window=3, min_periods=2).apply(
        lambda x: 1 if len(x[x > 0]) >= 2 or len(x[x < 0]) >= 2 else 0, raw=True
    )
    
    # Volume Flow Dynamics Analysis
    # Volume Distribution Patterns
    data['early_late_volume'] = (data['volume'].rolling(window=3, min_periods=2).sum() / 
                                data['volume'].rolling(window=10, min_periods=5).sum())
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).max()
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-Price Efficiency
    data['volume_weighted_price'] = (data['close'] - data['open']) * data['volume']
    data['price_impact_per_volume'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    
    # Extreme Volume Events
    data['volume_spike'] = (data['volume'] > data['volume'].rolling(window=20, min_periods=10).mean() * 2).astype(int)
    
    # Factor Construction Engine
    # Core Volatility-Adaptive Signal
    regime_momentums = []
    for idx, row in data.iterrows():
        if row['vol_regime'] == 2:  # High volatility
            momentum = row['high_vol_momentum']
        elif row['vol_regime'] == 1:  # Normal volatility
            momentum = row['normal_vol_momentum']
        else:  # Low volatility
            momentum = row['low_vol_momentum']
        regime_momentums.append(momentum)
    
    data['regime_weighted_momentum'] = regime_momentums
    
    # Volatility-adjusted scaling
    data['vol_adjusted_momentum'] = data['regime_weighted_momentum'] * data['vol_ratio']
    
    # Persistence enhancement
    data['persistence_enhanced'] = data['vol_adjusted_momentum'] * np.log1p(data['regime_duration'])
    
    # Volume Confirmation Mechanism
    data['volume_flow_alignment'] = data['volume_momentum'] * data['regime_weighted_momentum']
    
    # Efficiency filter
    data['efficiency_filtered'] = data['persistence_enhanced'] * (data['price_efficiency'] > 0.3)
    
    # Extreme event adjustment
    data['extreme_adjusted'] = data['efficiency_filtered'] * (1 + 0.2 * data['volume_spike'])
    
    # Multi-Timeframe Integration
    # Short-term confirmation (intraday patterns)
    data['intraday_strength'] = (data['close'] - data['open']) / data['true_range']
    
    # Medium-term validation (5-day consistency)
    data['medium_term_validation'] = data['normal_vol_momentum'].rolling(window=3, min_periods=2).mean()
    
    # Signal Refinement Process
    # Quality scoring
    data['momentum_quality'] = (data['price_efficiency'] + data['direction_consistency']) / 2
    data['volume_efficiency'] = data['early_late_volume'] * (1 - data['volume_concentration'])
    data['quality_score'] = (data['momentum_quality'] + data['volume_efficiency']) / 2
    
    # Consistency checks
    data['multi_timeframe_alignment'] = (
        (np.sign(data['intraday_strength']) == np.sign(data['medium_term_validation'])).astype(int)
    )
    
    # Final Factor Generation
    data['final_factor'] = (
        data['extreme_adjusted'] * 
        data['quality_score'] * 
        (1 + 0.1 * data['multi_timeframe_alignment']) *
        (1 + 0.05 * np.sign(data['vol_momentum']))
    )
    
    # Handle NaN values
    data['final_factor'] = data['final_factor'].fillna(0)
    
    return data['final_factor']
