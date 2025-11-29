import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Raw Momentum Strength
    # Daily momentum intensity
    data['momentum_intensity'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-day momentum persistence
    data['price_change'] = data['close'] - data['open']
    data['direction'] = np.sign(data['price_change'])
    data['consecutive_days'] = 0
    
    for i in range(1, len(data)):
        if data['direction'].iloc[i] == data['direction'].iloc[i-1] and data['direction'].iloc[i] != 0:
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
    
    # Directional Volume Concentration
    # Bullish volume ratio
    bullish_mask = data['close'] > data['open']
    data['bullish_volume_ratio'] = data['volume'].where(bullish_mask, 0) / (data['volume'] + 1e-8)
    
    # Bearish volume ratio
    bearish_mask = data['close'] < data['open']
    data['bearish_volume_ratio'] = data['volume'].where(bearish_mask, 0) / (data['volume'] + 1e-8)
    
    # Volume symmetry index
    data['volume_symmetry'] = abs(data['bullish_volume_ratio'] - data['bearish_volume_ratio'])
    
    # Price-Volume Divergence
    # Volume-adjusted momentum
    data['volume_adjusted_momentum'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    
    # Momentum-volume consistency analysis
    data['momentum_volume_consistency'] = np.where(
        (data['price_change'] > 0) & (data['bullish_volume_ratio'] > 0.5), 1,
        np.where((data['price_change'] < 0) & (data['bearish_volume_ratio'] > 0.5), -1, 0)
    )
    
    # Divergence between momentum strength and volume support
    data['momentum_volume_divergence'] = data['momentum_intensity'] * (1 - data['volume_symmetry'])
    
    # Microstructural Flow Patterns
    # Opening momentum efficiency
    data['prev_close'] = data['close'].shift(1)
    data['opening_efficiency'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    
    # Closing volume concentration (approximated using last 30 minutes volume pattern)
    # Using rolling window to estimate closing concentration
    data['volume_rolling_30min'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['closing_concentration'] = data['volume'] / (data['volume_rolling_30min'] + 1e-8)
    
    # Range-Momentum Elasticity
    # Volume per unit momentum
    data['volume_per_momentum'] = data['volume'] / (abs(data['close'] - data['open']) + 1e-8)
    
    # Momentum expansion efficiency
    data['momentum_expansion_efficiency'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    
    # Extreme Event Analysis
    # Gap volume intensity
    data['gap'] = abs(data['open'] - data['prev_close'])
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['gap_volume_intensity'] = np.where(
        data['gap'] > data['gap'].rolling(window=10, min_periods=1).mean(),
        data['volume'] / (data['avg_volume_5d'] + 1e-8),
        0
    )
    
    # Volume concentration during momentum breakouts
    data['momentum_breakout'] = (data['momentum_intensity'].abs() > 
                                data['momentum_intensity'].abs().rolling(window=10, min_periods=1).mean() + 
                                data['momentum_intensity'].abs().rolling(window=10, min_periods=1).std())
    data['breakout_volume_concentration'] = np.where(
        data['momentum_breakout'],
        data['volume'] / (data['avg_volume_5d'] + 1e-8),
        0
    )
    
    # Composite Scoring
    # Normalize components
    components = [
        'momentum_intensity',
        'consecutive_days',
        'volume_symmetry',
        'volume_adjusted_momentum',
        'momentum_volume_divergence',
        'opening_efficiency',
        'closing_concentration',
        'volume_per_momentum',
        'momentum_expansion_efficiency',
        'gap_volume_intensity',
        'breakout_volume_concentration'
    ]
    
    # Calculate rolling z-scores for normalization
    composite_score = pd.Series(0, index=data.index)
    
    for component in components:
        if component in data.columns:
            # Use rolling mean and std for normalization (20-day window)
            rolling_mean = data[component].rolling(window=20, min_periods=10).mean()
            rolling_std = data[component].rolling(window=20, min_periods=10).std()
            normalized_component = (data[component] - rolling_mean) / (rolling_std + 1e-8)
            
            # Weight components based on their expected predictive power
            if component in ['momentum_intensity', 'momentum_volume_divergence']:
                weight = 0.15
            elif component in ['volume_symmetry', 'volume_adjusted_momentum']:
                weight = 0.12
            elif component in ['consecutive_days', 'opening_efficiency']:
                weight = 0.10
            else:
                weight = 0.08
                
            composite_score += normalized_component * weight
    
    # Clean up intermediate columns
    drop_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    data = data.drop(columns=drop_cols)
    
    return composite_score
