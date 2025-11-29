import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    data['range'] = (data['high'] - data['low']) / data['close']
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Momentum components
    data['momentum_1d'] = data['intraday_return']
    data['momentum_1d_prev'] = data['momentum_1d'].shift(1)
    data['momentum_divergence'] = np.sign(data['momentum_1d']) != np.sign(data['momentum_1d_prev'])
    
    # Volatility components
    data['volatility_1d'] = data['range']
    data['volatility_1d_prev'] = data['volatility_1d'].shift(1)
    data['volatility_3d_avg'] = data['range'].rolling(window=3, min_periods=1).mean()
    
    # Volume features
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_spike'] = data['volume_ratio'] > 1.5
    
    # Gap features
    data['gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_capture'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    # Multi-day momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Compression features
    data['range_3d_avg'] = data['range'].rolling(window=3, min_periods=1).mean()
    data['range_compression'] = data['range'] / data['range_3d_avg']
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Initialize factor components
    momentum_component = np.zeros(len(data))
    efficiency_component = np.zeros(len(data))
    compression_component = np.zeros(len(data))
    transition_component = np.zeros(len(data))
    
    for i in range(1, len(data)):
        # Intraday Momentum-Volatility Divergence
        if not pd.isna(data['momentum_1d'].iloc[i]) and not pd.isna(data['momentum_1d_prev'].iloc[i]):
            momentum_dir = np.sign(data['momentum_1d'].iloc[i])
            prev_momentum_dir = np.sign(data['momentum_1d_prev'].iloc[i])
            efficiency = data['efficiency'].iloc[i] if not pd.isna(data['efficiency'].iloc[i]) else 0
            
            if momentum_dir == prev_momentum_dir and efficiency > 0.6:
                # High efficiency same direction
                momentum_component[i] = data['momentum_1d'].iloc[i] * efficiency * (1 + data['volume_ratio'].iloc[i])
            elif momentum_dir != prev_momentum_dir and efficiency < 0.4:
                # Low efficiency opposite direction (reversal)
                momentum_component[i] = -data['momentum_1d'].iloc[i] * (1 - efficiency) * (1 + data['volume_ratio'].iloc[i])
        
        # Volatility-Weighted Gap Capture Efficiency
        if not pd.isna(data['gap'].iloc[i]) and abs(data['gap'].iloc[i]) > 0.005:
            gap_dir = np.sign(data['gap'].iloc[i])
            capture_efficiency = data['gap_capture'].iloc[i] if not pd.isna(data['gap_capture'].iloc[i]) else 0
            
            if gap_dir * capture_efficiency > 0.3:  # Efficient gap capture
                vol_adjustment = 1 / (1 + data['volatility_1d'].iloc[i])
                efficiency_component[i] = capture_efficiency * vol_adjustment * data['volume_ratio'].iloc[i]
        
        # Compression Breakout Momentum
        if not pd.isna(data['range_compression'].iloc[i]) and data['range_compression'].iloc[i] < 0.8:
            # Compression detected
            breakout_strength = data['intraday_return'].iloc[i] / (data['range'].iloc[i] + 0.001)
            volume_confirmation = 1 if data['volume_ratio'].iloc[i] > 1.2 else 0.5
            
            if abs(breakout_strength) > 1.0:
                compression_component[i] = breakout_strength * volume_confirmation * (1 / data['range_compression'].iloc[i])
        
        # Regime Transition Efficiency
        if not pd.isna(data['momentum_3d'].iloc[i]) and not pd.isna(data['momentum_5d'].iloc[i]):
            mom_3d_dir = np.sign(data['momentum_3d'].iloc[i])
            mom_5d_dir = np.sign(data['momentum_5d'].iloc[i])
            
            if mom_3d_dir != mom_5d_dir:  # Potential regime transition
                crossover_magnitude = abs(data['momentum_3d'].iloc[i] - data['momentum_5d'].iloc[i])
                volume_confirmation = min(data['volume_ratio'].iloc[i], 2.0)
                vol_adjustment = 1 / (1 + data['volatility_1d'].iloc[i])
                
                transition_component[i] = crossover_magnitude * volume_confirmation * vol_adjustment * mom_3d_dir
    
    # Combine components with weights
    factor = (
        0.35 * pd.Series(momentum_component, index=data.index) +
        0.25 * pd.Series(efficiency_component, index=data.index) +
        0.20 * pd.Series(compression_component, index=data.index) +
        0.20 * pd.Series(transition_component, index=data.index)
    )
    
    # Apply rolling normalization
    factor_std = factor.rolling(window=20, min_periods=1).std().replace(0, 1)
    factor_mean = factor.rolling(window=20, min_periods=1).mean()
    normalized_factor = (factor - factor_mean) / factor_std
    
    return normalized_factor
