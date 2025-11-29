import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Intraday Momentum Structure
    # Opening Momentum: Open / Previous Close
    data['prev_close'] = data['close'].shift(1)
    data['opening_momentum'] = data['open'] / data['prev_close']
    
    # Intraday Momentum: Close / Open
    data['intraday_momentum'] = data['close'] / data['open']
    
    # Momentum Persistence: Count consecutive same-sign intraday momentum over 3 days
    data['intraday_momentum_sign'] = np.sign(data['intraday_momentum'] - 1)
    data['momentum_persistence'] = 0
    
    for i in range(2, len(data)):
        current_sign = data['intraday_momentum_sign'].iloc[i]
        prev1_sign = data['intraday_momentum_sign'].iloc[i-1]
        prev2_sign = data['intraday_momentum_sign'].iloc[i-2]
        
        if current_sign == prev1_sign == prev2_sign:
            data.loc[data.index[i], 'momentum_persistence'] = 3
        elif current_sign == prev1_sign:
            data.loc[data.index[i], 'momentum_persistence'] = 2
        else:
            data.loc[data.index[i], 'momentum_persistence'] = 1
    
    # Volatility Compression Pattern
    # Current Range: High - Low
    data['current_range'] = data['high'] - data['low']
    
    # Expected Range: 3-day average of (High - Low)
    data['expected_range'] = data['current_range'].rolling(window=3, min_periods=1).mean()
    
    # Compression Ratio: Current Range / Expected Range
    data['compression_ratio'] = data['current_range'] / data['expected_range']
    
    # Liquidity-Enhanced Reversal
    # Previous Day's Price Change: Close / Close from 2 days ago
    data['close_2d_ago'] = data['close'].shift(2)
    data['prev_day_price_change'] = data['close'].shift(1) / data['close_2d_ago']
    
    # Volume Ratio: Volume / 5-day average of Volume
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'].shift(1) / data['volume_5d_avg'].shift(1)
    
    # Reversal Signal: Previous Day's Price Change * Volume Ratio
    data['reversal_signal'] = data['prev_day_price_change'] * data['volume_ratio']
    
    # Combined Factor Construction
    # Momentum-Compression Composite: Intraday Momentum * Momentum Persistence * Compression Ratio
    data['momentum_compression_composite'] = (
        data['intraday_momentum'] * 
        data['momentum_persistence'] * 
        data['compression_ratio']
    )
    
    # Liquidity-Filtered Reversal: Reversal Signal * (Volume / Amount)
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    data['liquidity_filtered_reversal'] = data['reversal_signal'] * data['volume_amount_ratio']
    
    # Final Factor: Momentum-Compression Composite * Liquidity-Filtered Reversal
    data['final_factor'] = data['momentum_compression_composite'] * data['liquidity_filtered_reversal']
    
    # Volume-Volatility Alignment
    # Volume-Volatility Interaction: (High - Low) * Volume
    data['volume_volatility_interaction'] = data['current_range'] * data['volume']
    
    # Historical Context: Volume-Volatility Interaction / 3-day average of Volume-Volatility Interaction
    data['volume_volatility_3d_avg'] = data['volume_volatility_interaction'].rolling(window=3, min_periods=1).mean()
    data['historical_context'] = data['volume_volatility_interaction'] / data['volume_volatility_3d_avg']
    
    # Enhanced Factor: Final Factor * Historical Context
    data['enhanced_factor'] = data['final_factor'] * data['historical_context']
    
    # Return the enhanced factor series
    return data['enhanced_factor']
