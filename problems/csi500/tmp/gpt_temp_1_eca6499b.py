import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    data['high_low_range'] = data['high'] - data['low']
    data['open_low_range'] = data['open'] - data['low']
    data['close_open_diff'] = data['close'] - data['open']
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Calculate momentum components
    data['opening_momentum'] = (data['high'] - data['open']) / np.maximum(data['open'] - data['low'], 1e-8)
    data['opening_efficiency'] = np.abs(data['close'] - data['open']) / np.maximum(data['high_low_range'], 1e-8)
    data['midday_momentum'] = data['mid_price'] - data['open']
    data['closing_momentum'] = data['close'] - data['mid_price']
    data['momentum_decay_ratio'] = data['closing_momentum'] / np.maximum(data['opening_momentum'], 1e-8)
    
    # Calculate volume-amount components
    data['amount_per_volume'] = data['amount'] / np.maximum(data['volume'], 1e-8)
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['amount_change'] = data['amount'] - data['amount'].shift(1)
    data['volume_amount_divergence'] = np.sign(data['volume_change']) * np.sign(data['amount_change'])
    
    # Calculate range compression components
    data['current_range'] = data['high'] - data['low']
    data['range_3d_ago'] = data['high'].shift(3) - data['low'].shift(3)
    data['range_compression'] = data['current_range'] / np.maximum(data['range_3d_ago'], 1e-8)
    
    data['opening_position'] = (data['open'] - data['low']) / np.maximum(data['high_low_range'], 1e-8)
    data['closing_position'] = (data['close'] - data['low']) / np.maximum(data['high_low_range'], 1e-8)
    data['position_shift'] = data['closing_position'] - data['opening_position']
    
    # Calculate accumulation signals
    data['large_trade_ratio'] = data['amount'] / np.maximum(data['volume'] * data['close'], 1e-8)
    data['amount_persistence'] = data['amount'] / np.maximum(data['amount'].shift(1), 1e-8)
    
    vwp = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    awp = (data['high'] + data['low'] + data['close']) / 3 * data['amount']
    data['accumulation_signal'] = awp / np.maximum(vwp, 1e-8)
    
    # Calculate efficiency components
    data['range_utilization'] = np.abs(data['close'] - data['open']) / np.maximum(data['high_low_range'], 1e-8)
    data['directional_efficiency'] = (data['close'] - data['open']) / np.maximum(data['high_low_range'], 1e-8)
    
    data['volume_range_ratio'] = data['volume'] / np.maximum(data['high_low_range'], 1e-8)
    data['volume_spike_persistence'] = data['volume'] / np.maximum(data['volume'].shift(1), 1e-8)
    data['divergence_signal'] = np.sign(data['close'] - data['open']) * np.sign(data['volume_change'])
    
    # Calculate trade quality
    data['amount_per_trade'] = data['amount'] / np.maximum(data['volume'], 1e-8)
    data['trade_quality_change'] = data['amount_per_trade'] / np.maximum(data['amount_per_trade'].shift(1), 1e-8)
    
    data['price_amount_efficiency'] = (data['close'] - data['open']) * data['amount']
    data['volume_amount_correlation'] = np.sign(data['volume_change']) * np.sign(data['amount_change'])
    data['microstructure_alignment'] = data['price_amount_efficiency'] * data['volume_amount_correlation']
    
    # Calculate lagged components for persistence
    data['prev_day_pattern'] = (data['close'].shift(1) - data['open'].shift(1)) / np.maximum(data['high'].shift(1) - data['low'].shift(1), 1e-8)
    data['decay_persistence'] = data['momentum_decay_ratio'] / np.maximum(data['momentum_decay_ratio'].shift(1), 1e-8)
    
    data['prev_volume_change'] = data['volume'].shift(1) - data['volume'].shift(2)
    data['prev_divergence'] = np.sign(data['close'].shift(1) - data['open'].shift(1)) * np.sign(data['prev_volume_change'])
    data['divergence_persistence'] = data['divergence_signal'] / np.maximum(data['prev_divergence'], 1e-8)
    
    # Calculate breakout timing components
    data['pre_breakout_volume'] = data['volume'] / np.maximum(data['volume'].shift(1), 1e-8)
    data['breakout_imminence'] = 1 / np.maximum(data['range_compression'], 1e-8)
    
    # Combine factors from different strategies
    for i in range(len(data)):
        if i < 4:  # Need at least 4 days of history
            factor.iloc[i] = 0
            continue
            
        # Strategy 1: Momentum Decay with Volume-Amount Divergence
        base_decay_signal = data['momentum_decay_ratio'].iloc[i] * data['opening_efficiency'].iloc[i]
        enhanced_decay = base_decay_signal * data['volume_amount_divergence'].iloc[i]
        factor1 = enhanced_decay * data['prev_day_pattern'].iloc[i] * data['decay_persistence'].iloc[i]
        
        # Strategy 2: Range Compression with Breakout Anticipation
        base_compression = data['range_compression'].iloc[i] * data['position_shift'].iloc[i]
        enhanced_compression = base_compression * data['accumulation_signal'].iloc[i]
        factor2 = enhanced_compression * data['pre_breakout_volume'].iloc[i] * data['breakout_imminence'].iloc[i]
        
        # Strategy 3: Volume-Price Divergence with Range Efficiency
        base_divergence = data['divergence_signal'].iloc[i] * data['range_utilization'].iloc[i]
        enhanced_divergence = base_divergence * data['microstructure_alignment'].iloc[i]
        factor3 = enhanced_divergence * data['prev_divergence'].iloc[i] * data['divergence_persistence'].iloc[i]
        
        # Combine strategies with equal weighting
        factor.iloc[i] = (factor1 + factor2 + factor3) / 3
    
    # Handle any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
