import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate returns and basic metrics
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    
    # 1. Microstructural Momentum Regimes
    # Calculate bidirectional price pressure using 5-minute intervals (approximated)
    data['up_move'] = (data['returns'] > 0).astype(int)
    data['down_move'] = (data['returns'] < 0).astype(int)
    
    # Calculate consecutive moves
    data['consecutive_up'] = data['up_move'] * (data['up_move'].groupby(data.index).cumcount() + 1)
    data['consecutive_down'] = data['down_move'] * (data['down_move'].groupby(data.index).cumcount() + 1)
    
    # Net microstructural pressure
    data['up_pressure'] = data['consecutive_up'] * data['returns'].clip(lower=0)
    data['down_pressure'] = data['consecutive_down'] * abs(data['returns'].clip(upper=0))
    data['net_pressure'] = (data['up_pressure'] - data['down_pressure']).rolling(window=12, min_periods=1).mean()
    
    # Momentum decay characteristics
    data['move_duration'] = np.maximum(data['consecutive_up'], data['consecutive_down'])
    data['duration_persistence'] = data['move_duration'].rolling(window=6, min_periods=1).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, raw=True
    )
    
    # Return compression patterns
    data['return_compression'] = data['returns'].rolling(window=6, min_periods=1).apply(
        lambda x: x[-1] / x[0] if abs(x[0]) > 0.001 and len(x) > 1 else 1, raw=True
    )
    
    # Momentum decay score
    data['momentum_decay_score'] = (data['duration_persistence'] * data['return_compression']).fillna(0)
    data['momentum_decay_score'] = 1 / (1 + abs(data['momentum_decay_score']))
    data['momentum_decay_score'] = data['momentum_decay_score'] / data['high_low_range'].rolling(window=6, min_periods=1).std().replace(0, 1)
    
    # 2. Volume-Price Fractal Dynamics
    # Multi-timeframe volume clustering
    data['volume_zscore_15min'] = (data['volume'] - data['volume'].rolling(window=3, min_periods=1).mean()) / data['volume'].rolling(window=3, min_periods=1).std().replace(0, 1)
    data['volume_burst'] = (data['volume_zscore_15min'] > 2).astype(int)
    data['burst_frequency'] = data['volume_burst'].rolling(window=12, min_periods=1).mean()
    
    # Volume persistence
    data['high_volume'] = (data['volume'] > data['volume'].rolling(window=24, min_periods=1).median()).astype(int)
    data['volume_persistence'] = data['high_volume'].rolling(window=4, min_periods=1).mean()
    
    # Volume fractal dimension
    data['volume_fractal'] = (data['burst_frequency'] / (data['volume_persistence'] + 0.001)).apply(np.log1p)
    
    # Price movement self-similarity
    data['return_variance_ratio'] = data['returns'].rolling(window=3, min_periods=1).std() / data['returns'].rolling(window=12, min_periods=1).std().replace(0, 1)
    
    # Price path complexity
    data['directional_changes'] = (data['returns'].diff() != 0).astype(int).rolling(window=12, min_periods=1).sum()
    data['path_efficiency'] = abs(data['close'].diff(12)) / (data['high'] - data['low']).rolling(window=12, min_periods=1).sum().replace(0, 1)
    
    # Price fractal score
    data['price_fractal_score'] = data['return_variance_ratio'] * data['path_efficiency']
    data['price_fractal_score'] = data['price_fractal_score'] / data['high_low_range'].rolling(window=12, min_periods=1).mean().replace(0, 1)
    
    # Combined volume-price fractal
    data['fractal_component'] = data['volume_fractal'] * data['price_fractal_score']
    
    # 3. Order Flow Imbalance Acceleration
    # Volume-weighted price pressure
    data['volume_pressure'] = data['volume'] * data['returns']
    data['positive_pressure'] = data['volume_pressure'].clip(lower=0).rolling(window=6, min_periods=1).sum()
    data['negative_pressure'] = abs(data['volume_pressure'].clip(upper=0)).rolling(window=6, min_periods=1).sum()
    
    # Imbalance momentum
    data['net_imbalance'] = data['positive_pressure'] - data['negative_pressure']
    data['imbalance_momentum'] = data['net_imbalance'].diff(3) / (data['volume'].rolling(window=6, min_periods=1).sum().replace(0, 1))
    
    # Imbalance exhaustion signals
    data['cumulative_imbalance'] = data['volume_pressure'].rolling(window=24, min_periods=1).sum()
    data['imbalance_extreme'] = (abs(data['cumulative_imbalance']) > data['cumulative_imbalance'].rolling(window=48, min_periods=1).std() * 2).astype(int)
    
    # Imbalance mean reversion
    data['imbalance_reversion'] = data['net_imbalance'].rolling(window=6, min_periods=1).apply(
        lambda x: -np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, raw=True
    )
    
    # Exhaustion score
    data['exhaustion_score'] = data['imbalance_extreme'] * data['imbalance_reversion']
    data['exhaustion_score'] = 1 / (1 + abs(data['exhaustion_score']))
    
    # Combined order flow component
    data['flow_component'] = data['imbalance_momentum'] * data['exhaustion_score']
    
    # 4. Price-Level Memory Effects
    # Price anchor points using volume concentration
    data['price_level'] = (data['close'] / 0.01).round() * 0.01  # Round to nearest penny
    data['volume_at_level'] = data.groupby('price_level')['volume'].transform('cumsum')
    
    # Anchor proximity score
    recent_anchors = data.groupby('price_level')['volume'].rolling(window=24, min_periods=1).sum().groupby('price_level').tail(1)
    if not recent_anchors.empty:
        strongest_anchor = recent_anchors.idxmax()[0] if isinstance(recent_anchors.idxmax(), tuple) else recent_anchors.idxmax()
        data['anchor_distance'] = abs(data['close'] - strongest_anchor) / data['close']
    else:
        data['anchor_distance'] = 0
    
    data['anchor_proximity'] = 1 / (1 + data['anchor_distance'])
    
    # Memory-driven reversal patterns
    data['reversal_strength'] = -data['returns'].rolling(window=6, min_periods=1).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, raw=True
    )
    
    # Memory effect signal
    data['memory_component'] = data['anchor_proximity'] * data['reversal_strength']
    data['memory_component'] = data['memory_component'] / data['high_low_range'].rolling(window=12, min_periods=1).mean().replace(0, 1)
    
    # 5. Synthesize Multi-Regime Alpha Factor
    # Calculate regime alignment
    components = ['momentum_decay_score', 'fractal_component', 'flow_component', 'memory_component']
    data['regime_congruence'] = data[components].std(axis=1) / (data[components].mean(axis=1).abs() + 0.001)
    data['regime_congruence'] = 1 / (1 + data['regime_congruence'])
    
    # Dynamic component weights based on regime characteristics
    data['momentum_weight'] = 1 - abs(data['momentum_decay_score'])
    data['fractal_weight'] = abs(data['fractal_component'])
    data['flow_weight'] = abs(data['flow_component'])
    data['memory_weight'] = data['anchor_proximity']
    
    # Normalize weights
    total_weight = data[['momentum_weight', 'fractal_weight', 'flow_weight', 'memory_weight']].sum(axis=1).replace(0, 1)
    for weight_col in ['momentum_weight', 'fractal_weight', 'flow_weight', 'memory_weight']:
        data[weight_col] = data[weight_col] / total_weight
    
    # Composite alpha signal
    data['composite_alpha'] = (
        data['momentum_weight'] * data['momentum_decay_score'] +
        data['fractal_weight'] * data['fractal_component'] +
        data['flow_weight'] * data['flow_component'] +
        data['memory_weight'] * data['memory_component']
    )
    
    # Apply regime congruence and final enhancements
    data['alpha_factor'] = data['composite_alpha'] * data['regime_congruence']
    
    # Apply momentum-regime specific filter
    data['alpha_factor'] = data['alpha_factor'] * (1 - abs(data['momentum_decay_score']))
    
    # Final signal persistence
    data['alpha_factor'] = data['alpha_factor'].rolling(window=6, min_periods=1).mean()
    
    return data['alpha_factor']
