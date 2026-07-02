import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    for t in range(len(data)):
        current_data = data.iloc[:t+1]  # Only use data up to current time t
        
        # Calculate midpoint momentum
        midpoint = (current_data['high'] + current_data['low']) / 2
        midpoint_momentum = (current_data['close'] - midpoint) / midpoint
        
        # Normalize by daily range
        daily_range = (current_data['high'] - current_data['low']) / current_data['close']
        normalized_momentum = midpoint_momentum / daily_range
        
        # Calculate volume range
        if t >= 20:  # Need at least 20 days for median calculation
            median_volume = current_data['volume'].rolling(20).median().iloc[-1]
            volume_range = current_data['volume'].iloc[-1] / median_volume
        else:
            volume_range = 1.0  # Default value when insufficient history
            
        # Adjust momentum by volume range
        volume_adjusted_momentum = normalized_momentum.iloc[-1] * volume_range
        
        # Calculate morning and afternoon momentum if intraday data available
        if 'open' in current_data.columns and 'close' in current_data.columns:
            # Assuming we can identify midday high/low from intraday data
            # For simplicity, we'll use high/low up to current time as proxy
            morning_momentum = (current_data['high'].iloc[-1] - current_data['open'].iloc[-1]) / current_data['open'].iloc[-1]
            afternoon_momentum = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / current_data['low'].iloc[-1]
            
            # Calculate momentum divergence
            momentum_divergence = morning_momentum - afternoon_momentum
        else:
            momentum_divergence = 0.0
            
        # Combine all components
        factor.iloc[t] = volume_adjusted_momentum * (1 + momentum_divergence)
    
    return factor
