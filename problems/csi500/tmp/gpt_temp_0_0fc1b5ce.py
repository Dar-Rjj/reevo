import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Range-Gap Momentum Factor
    Combines intraday range dynamics with volume-weighted gap analysis
    to identify momentum patterns with volume validation.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate Range Momentum (5-day rolling)
    data['range_momentum'] = data['true_range'] / data['true_range'].rolling(window=5, min_periods=3).mean() - 1
    
    # Calculate Volume-Weighted Close-to-Open Return
    data['close_to_open_return'] = data['close'] / data['open'] - 1
    data['volume_weighted_gap'] = data['close_to_open_return'] * data['volume']
    
    # Normalize volume-weighted gap by rolling standard deviation (10-day window)
    data['norm_volume_gap'] = data['volume_weighted_gap'] / data['volume_weighted_gap'].rolling(window=10, min_periods=5).std()
    
    # Calculate intraday price position relative to range
    data['range_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Position momentum (change in position from previous day)
    data['position_momentum'] = data['range_position'] - data['range_position'].shift(1)
    
    # Volume validation metrics
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Multi-timeframe volatility compression detection
    data['volatility_ratio'] = data['true_range'] / data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Core factor calculation combining all components
    factor = np.zeros(len(data))
    
    for i in range(2, len(data)):
        # Skip if insufficient data
        if any(pd.isna(data.iloc[i][['true_range', 'range_momentum', 'norm_volume_gap', 'range_position', 'volume_ratio']])):
            factor[i] = 0
            continue
            
        current_data = data.iloc[i]
        prev_data = data.iloc[i-1]
        
        # Range-Gap Momentum Component
        range_gap_momentum = 0
        
        # Large gap with contracting range (potential reversal)
        if abs(current_data['norm_volume_gap']) > 1.5 and current_data['range_momentum'] < -0.1:
            range_gap_momentum = -current_data['norm_volume_gap'] * 0.7
            
        # Small gap with expanding range (continuation)
        elif abs(current_data['norm_volume_gap']) < 0.5 and current_data['range_momentum'] > 0.1:
            range_gap_momentum = current_data['range_momentum'] * 1.2
            
        # Volume-Validated Range-Gap Alignment
        if current_data['volume_ratio'] > 1.2:
            # High volume with expanding range and positive gap
            if current_data['range_momentum'] > 0.1 and current_data['norm_volume_gap'] > 0.5:
                range_gap_momentum += current_data['norm_volume_gap'] * 0.8
            # High volume with contracting range and negative gap
            elif current_data['range_momentum'] < -0.1 and current_data['norm_volume_gap'] < -0.5:
                range_gap_momentum += current_data['norm_volume_gap'] * 0.8
        
        # Volatility Compression Breakout Component
        volatility_component = 0
        
        # Low volatility environment
        if current_data['volatility_ratio'] < 0.7:
            # Large volume-weighted gap as potential breakout trigger
            if abs(current_data['norm_volume_gap']) > 1.0:
                volatility_component = current_data['norm_volume_gap'] * 1.5
        
        # Range-Position Momentum Component
        position_component = 0
        
        # Position momentum with volume-weighted gap alignment
        if abs(current_data['position_momentum']) > 0.1:
            # Volume-weighted gap supporting position trend
            if (current_data['position_momentum'] > 0 and current_data['norm_volume_gap'] > 0) or \
               (current_data['position_momentum'] < 0 and current_data['norm_volume_gap'] < 0):
                position_component = current_data['position_momentum'] * current_data['norm_volume_gap'] * 2.0
        
        # Combine all components with volume validation
        combined_factor = (
            range_gap_momentum * 0.4 +
            volatility_component * 0.3 +
            position_component * 0.3
        ) * np.log1p(current_data['volume_ratio'])
        
        factor[i] = combined_factor
    
    # Create output series
    factor_series = pd.Series(factor, index=data.index)
    
    # Remove any remaining NaN values
    factor_series = factor_series.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
