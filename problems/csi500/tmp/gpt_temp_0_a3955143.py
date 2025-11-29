import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Scale Momentum Asymmetry Efficiency Factor
    Combines momentum efficiency, volume asymmetry, and microstructural flow patterns
    """
    data = df.copy()
    
    # Multi-Scale Momentum Efficiency Analysis
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['overnight_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Cross-scale efficiency divergence patterns
    data['efficiency_alignment'] = np.sign(data['intraday_efficiency']) * np.sign(data['overnight_efficiency'])
    data['efficiency_strength'] = (abs(data['intraday_efficiency']) + abs(data['overnight_efficiency'])) / 2
    
    # Volume Asymmetry Confirmation Dynamics
    data['bullish_volume_ratio'] = np.where(data['close'] > data['open'], data['volume'], 0) / data['volume']
    data['bearish_volume_ratio'] = np.where(data['close'] < data['open'], data['volume'], 0) / data['volume']
    data['volume_symmetry'] = abs(data['bullish_volume_ratio'] - data['bearish_volume_ratio'])
    
    # Volume absorption intensity
    data['midpoint_volume'] = np.where(
        (data['high'] + data['low']) / 2 >= data['close'],
        data['volume'], 0
    )
    data['volume_absorption'] = data['midpoint_volume'] / data['volume'].rolling(5).mean()
    
    # Momentum-Volume Divergence Detection
    data['price_change_magnitude'] = abs(data['close'] - data['close'].shift(1))
    data['volume_adjusted_momentum'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    
    # Directional consistency validation
    data['momentum_direction'] = np.sign(data['close'] - data['open'])
    data['consecutive_momentum'] = data['momentum_direction'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    ).reset_index(level=0, drop=True)
    
    # Microstructural Flow Efficiency Integration
    # Using rolling windows as proxy for session-based analysis
    data['opening_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Range-Efficiency Elasticity Framework
    data['volume_per_efficiency'] = data['volume'] / abs(data['close'] - data['open']).replace(0, np.nan)
    data['efficiency_elasticity'] = (data['high'] - data['low']) / data['volume'].replace(0, np.nan)
    
    # Extreme Event Efficiency Asymmetry
    data['gap_volume_intensity'] = data['volume'] / data['volume'].rolling(10).mean()
    data['gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Cross-Scale Asymmetry-Efficiency Scoring
    # Component 1: Efficiency-Volume Alignment
    efficiency_volume_score = (
        data['efficiency_strength'].rolling(3).mean() * 
        data['volume_symmetry'].rolling(3).mean() * 
        np.sign(data['intraday_efficiency'])
    )
    
    # Component 2: Momentum-Volume Persistence
    momentum_persistence_score = (
        data['consecutive_momentum'] * 
        data['volume_absorption'].rolling(3).mean() * 
        data['momentum_direction']
    )
    
    # Component 3: Elasticity-Flow Integration
    elasticity_score = (
        (1 / data['efficiency_elasticity'].rolling(3).mean()) * 
        data['volume_per_efficiency'].rolling(3).mean() * 
        data['efficiency_alignment']
    )
    
    # Component 4: Extreme Event Efficiency
    extreme_score = (
        data['gap_volume_intensity'] * 
        data['gap_efficiency'] * 
        np.sign(data['open'] - data['close'].shift(1))
    )
    
    # Final Composite Factor
    factor = (
        0.4 * efficiency_volume_score +
        0.3 * momentum_persistence_score +
        0.2 * elasticity_score +
        0.1 * extreme_score
    )
    
    # Cross-sectional normalization
    factor = factor.groupby(factor.index).transform(lambda x: (x - x.mean()) / x.std())
    
    return factor.dropna()
