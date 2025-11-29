import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Gap-Compression Dynamics
    # Opening Gap Analysis
    data['gap'] = data['open'] - data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['gap_to_range_ratio'] = data['gap'] / data['daily_range']
    data['gap_closure_efficiency'] = (data['close'] - data['open']) / data['gap']
    
    # Range Compression Detection
    data['prev_range'] = data['prev_high'] - data['prev_low']
    data['current_range_ratio'] = data['daily_range'] / data['prev_range']
    
    # 5-day average range
    data['avg_5d_range'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['compression_ratio_5d'] = data['daily_range'] / data['avg_5d_range']
    
    # Volume-Regime Momentum
    # Volume Acceleration
    data['avg_3d_volume'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_acceleration'] = (data['volume'] - data['avg_3d_volume']) / data['avg_3d_volume']
    
    # Volume-to-Amplitude Ratio
    data['volume_to_amplitude'] = data['volume'] / data['daily_range']
    
    # Volume Regime Position (rank in 5-day window)
    data['volume_rank_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Breakout Efficiency Measurement
    # Intraday Absorption Ratio
    data['intraday_absorption'] = data['daily_range'] / abs(data['gap'])
    
    # Position Establishment
    data['position_establishment'] = data['amount'] * abs(data['close'] - data['open']) / data['daily_range']
    
    # Breakout Strength
    data['breakout_strength'] = (data['close'] - data['open']) * data['daily_range'] / data['avg_5d_range']
    
    # Momentum Signal Construction
    # 5-day return
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Compression-Adjusted Momentum
    data['compression_adj_momentum'] = data['return_5d'] * data['daily_range'] / data['avg_5d_range']
    
    # Gap-Enhanced Momentum
    data['gap_enhanced_momentum'] = data['compression_adj_momentum'] * abs(data['gap'])
    
    # Volume-Weighted Momentum
    data['volume_weighted_momentum'] = data['gap_enhanced_momentum'] * data['volume_acceleration']
    
    # Final Alpha Generation
    # Combine momentum components with volume confirmation
    momentum_component = data['volume_weighted_momentum'] * data['volume_rank_5d']
    
    # Adjust for absorption efficiency
    absorption_adjusted = momentum_component / (1 + data['intraday_absorption'])
    
    # Scale by compression state
    compression_scale = 1 / (1 + data['compression_ratio_5d'])
    
    # Final alpha factor
    alpha = absorption_adjusted * compression_scale
    
    # Clean infinite and NaN values
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    
    return alpha
