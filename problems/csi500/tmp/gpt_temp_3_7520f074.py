import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Harmony with Volume-Price Anchoring alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = data['close'] / data['prev_close'] - 1
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # Morning session momentum (first half of trading day approximation)
    data['morning_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Multi-timeframe momentum calculations
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum divergence and alignment
    data['momentum_divergence_3_10'] = data['momentum_3d'] - data['momentum_10d']
    data['momentum_alignment'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_10d'])
    
    # Momentum acceleration
    data['momentum_accel_3d'] = data['momentum_3d'] - data['momentum_3d'].shift(3)
    data['momentum_accel_5d'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    
    # Volume-price efficiency
    data['volume_efficiency'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    data['am_pm_volume_ratio'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # VWAP calculations
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
    
    # Range efficiency and position
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum path efficiency
    data['momentum_path_efficiency'] = abs(data['close'] - data['open']) / (data['daily_range'] * data['prev_close'] + 1e-8)
    
    # Volume-price divergence
    data['volume_price_divergence'] = (data['volume'] / data['volume'].rolling(window=10).mean()) - \
                                     (abs(data['daily_return']) / abs(data['daily_return']).rolling(window=10).mean())
    
    # Opening dynamics
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['opening_efficiency'] = abs(data['opening_gap']) / (data['daily_range'] + 1e-8)
    data['opening_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Multi-timeframe momentum consistency
    data['momentum_consistency'] = (data['momentum_3d'].rolling(window=5).std() + 1e-8) / \
                                  (abs(data['momentum_3d'].rolling(window=5).mean()) + 1e-8)
    
    # Volume-momentum harmony
    data['volume_momentum_correlation'] = data['volume'].rolling(window=5).corr(data['momentum_3d'])
    data['harmony_strength'] = data['volume_momentum_correlation'] * data['momentum_alignment']
    
    # Composite factor components
    # 1. Momentum-Volume Harmony Score
    momentum_volume_harmony = (
        data['harmony_strength'] * 
        (1 - data['momentum_consistency']) *  # Lower consistency is better for momentum
        np.sign(data['momentum_3d'])
    )
    
    # 2. Range Efficiency Score
    range_efficiency_score = (
        data['range_efficiency'] * 
        data['close_position'] * 
        (1 - abs(data['close_position'] - 0.5))  # Prefer positions away from middle
    )
    
    # 3. Volume Anchoring Score
    volume_anchoring_score = (
        data['vwap_deviation'] * 
        np.sign(data['momentum_3d']) * 
        (1 - abs(data['volume_price_divergence']))
    )
    
    # 4. Multi-timeframe Alignment Score
    timeframe_alignment_score = (
        data['momentum_alignment'] * 
        (1 - abs(data['momentum_divergence_3_10'])) * 
        np.tanh(data['momentum_accel_3d'] * 10)  # Scale acceleration
    )
    
    # 5. Opening Dynamics Score
    opening_score = (
        data['opening_gap'] * 
        data['opening_efficiency'] * 
        np.tanh(data['opening_volume_ratio'] - 1)
    )
    
    # Combine components with weights
    composite_factor = (
        0.35 * momentum_volume_harmony +
        0.25 * range_efficiency_score +
        0.20 * volume_anchoring_score +
        0.15 * timeframe_alignment_score +
        0.05 * opening_score
    )
    
    # Apply cross-sectional normalization
    def cross_sectional_rank(series):
        return series.rank(pct=True) - 0.5
    
    # Final factor with cross-sectional ranking
    final_factor = composite_factor.groupby(composite_factor.index).transform(cross_sectional_rank)
    
    # Clean up and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
