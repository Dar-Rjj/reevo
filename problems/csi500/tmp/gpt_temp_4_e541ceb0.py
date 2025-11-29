import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Calculate current range
    data['current_range'] = (data['high'] - data['low']) / data['open']
    
    # Calculate 5-day average range for volatility regime
    data['range_5d_avg'] = data['current_range'].rolling(window=5, min_periods=3).mean()
    
    # Calculate overnight gap
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Calculate turnover efficiency
    data['turnover_eff'] = abs(data['close'] - data['prev_close']) / data['amount']
    data['prev_turnover_eff'] = data['turnover_eff'].shift(1)
    
    # Calculate volume changes
    data['volume_change'] = data['volume'] - data['prev_volume']
    data['prev_volume_change'] = data['volume_change'].shift(1)
    data['volume_momentum'] = data['volume_change'] - data['prev_volume_change']
    
    # Calculate multi-day range for compression analysis
    data['high_4d_max'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_4d_min'] = data['low'].rolling(window=5, min_periods=3).min()
    data['multi_day_range'] = (data['high_4d_max'] - data['low_4d_min']) / data['open']
    
    # Calculate compression ratio
    data['compression_ratio'] = data['current_range'] / data['multi_day_range']
    
    # Calculate range expansion
    data['range_expansion'] = data['current_range'] / data['range_5d_avg']
    data['prev_range_expansion'] = data['range_expansion'].shift(1)
    data['expansion_momentum'] = data['range_expansion'] / data['prev_range_expansion']
    
    # Calculate gap persistence (simplified - using sign of gap and price action)
    data['gap_persistence'] = np.where(
        (data['overnight_gap'] > 0) & (data['low'] > data['prev_close']), 1,
        np.where((data['overnight_gap'] < 0) & (data['high'] < data['prev_close']), 1, 0)
    )
    data['gap_strength'] = abs(data['overnight_gap']) * data['gap_persistence']
    
    # Calculate intraday return and movement efficiency
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['movement_eff'] = abs(data['intraday_return']) / data['amount']
    
    # Calculate gap filling
    data['gap_filling'] = (data['close'] - data['open']) * np.sign(-data['overnight_gap'])
    data['filling_eff'] = abs(data['gap_filling']) / data['amount']
    
    # Calculate price position relative to previous day's range
    data['price_position'] = (data['close'] - data['prev_low']) / (data['prev_high'] - data['prev_low'])
    
    # Calculate regime indicator
    data['regime_indicator'] = data['current_range'] / data['range_5d_avg']
    data['regime_context'] = np.sign(data['regime_indicator'] - 1)
    
    # Calculate efficiency momentum
    data['efficiency_momentum'] = data['turnover_eff'] / data['prev_turnover_eff']
    
    # Calculate regime efficiency
    data['regime_efficiency'] = data['turnover_eff'] * data['regime_indicator']
    
    # Calculate breakout magnitude (simplified - using current day's range)
    data['breakout_magnitude'] = abs(data['close'] - data['high']) / data['current_range']
    data['breakout_eff'] = data['breakout_magnitude'] * data['movement_eff']
    
    # Calculate compression-breakout factor
    data['compression_breakout'] = data['compression_ratio'] * data['breakout_eff']
    
    # Calculate volume-aligned signal
    data['volume_aligned_signal'] = data['compression_breakout'] * data['volume_momentum']
    
    # Calculate range-weighted gap
    data['range_weighted_gap'] = data['gap_strength'] * data['range_expansion']
    
    # Calculate acceleration alignment
    data['acceleration_alignment'] = data['volume_momentum'] * np.sign(data['overnight_gap'])
    
    # Calculate volume-confirmed signal
    data['volume_confirmed_signal'] = data['range_weighted_gap'] * data['acceleration_alignment']
    
    # Calculate position-efficiency factor
    data['position_efficiency'] = data['price_position'] * data['regime_efficiency']
    
    # Calculate momentum-weighted signal
    data['momentum_weighted'] = data['position_efficiency'] * data['efficiency_momentum']
    
    # Calculate volume alignment
    data['volume_alignment'] = data['momentum_weighted'] * (data['volume'] / data['prev_volume'])
    
    # Combine signals with appropriate weights
    for idx in data.index:
        if pd.notna(data.loc[idx, 'volume_aligned_signal']) and pd.notna(data.loc[idx, 'volume_confirmed_signal']):
            # Weight the signals based on their characteristics
            signal1 = data.loc[idx, 'volume_aligned_signal']
            signal2 = data.loc[idx, 'volume_confirmed_signal']
            signal3 = data.loc[idx, 'volume_alignment']
            
            # Combine with emphasis on volume-confirmed signals
            combined_signal = 0.3 * signal1 + 0.4 * signal2 + 0.3 * signal3
            
            # Apply regime context adjustment
            if pd.notna(data.loc[idx, 'regime_context']):
                combined_signal *= (1 + 0.2 * data.loc[idx, 'regime_context'])
            
            result.loc[idx] = combined_signal
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
