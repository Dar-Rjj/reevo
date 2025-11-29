import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate all components for each day
    for i in range(len(data)):
        if i < 5:  # Need at least 5 days of history for some calculations
            factor.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # Intraday Range-Price Divergence Factor
        if (current['high'] - current['low']) != 0:
            movement_efficiency = (current['close'] - current['open']) / (current['high'] - current['low'])
        else:
            movement_efficiency = 0
            
        if i >= 5 and data.iloc[i-5]['close'] != 0 and data.iloc[i-5]['volume'] != 0:
            volume_price_alignment = (current['close'] / data.iloc[i-5]['close']) * (current['volume'] / data.iloc[i-5]['volume'])
        else:
            volume_price_alignment = 1
            
        # Volatility Adjustment - rolling 3-day average of high-low range
        if i >= 2:
            rolling_3_high_low = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(i-2, i+1)])
            if rolling_3_high_low != 0:
                volatility_adjustment = (current['high'] - current['low']) / rolling_3_high_low
            else:
                volatility_adjustment = 1
        else:
            volatility_adjustment = 1
            
        factor1 = movement_efficiency * volume_price_alignment * volatility_adjustment
        
        # Volume-Confirmed Opening Gap Momentum
        if data.iloc[i-1]['close'] != 0:
            gap_momentum = (current['open'] / data.iloc[i-1]['close'])
            # Count how many of last 3 days had opening gap > 1
            if i >= 3:
                gap_count = sum([1 for j in range(i-3, i) if data.iloc[j]['open'] / data.iloc[j-1]['close'] > 1])
                gap_momentum *= gap_count
        else:
            gap_momentum = 1
            
        if i >= 3 and data.iloc[i-3]['volume'] != 0:
            volume_momentum = (current['volume'] / data.iloc[i-3]['volume']) * ((current['close'] / current['open']) * current['volume'])
        else:
            volume_momentum = 1
            
        # Price Position - 5-day rolling average
        if i >= 4:
            rolling_5_close = np.mean([data.iloc[j]['close'] for j in range(i-4, i+1)])
            if rolling_5_close != 0:
                price_position = current['close'] / rolling_5_close
            else:
                price_position = 1
        else:
            price_position = 1
            
        factor2 = gap_momentum * volume_momentum * price_position
        
        # High-Low Compression Momentum Factor
        if i >= 2:
            rolling_3_hl = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(i-2, i+1)])
            if rolling_3_hl != 0:
                range_compression = (current['high'] - current['low']) / rolling_3_hl
            else:
                range_compression = 1
        else:
            range_compression = 1
            
        if i >= 5 and data.iloc[i-5]['close'] != 0 and data.iloc[i-2]['close'] != 0:
            momentum_acceleration = (current['close'] / data.iloc[i-2]['close']) / (current['close'] / data.iloc[i-5]['close'])
        else:
            momentum_acceleration = 1
            
        if i >= 3 and data.iloc[i-3]['volume'] != 0:
            volume_flow = (current['volume'] / data.iloc[i-3]['volume']) * ((current['close'] - current['open']) * current['volume'])
        else:
            volume_flow = 1
            
        factor3 = range_compression * momentum_acceleration * volume_flow
        
        # Open-Close Volatility Fractality Factor
        if i >= 4:
            rolling_2_hl = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(i-1, i+1)])
            rolling_5_hl = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(i-4, i+1)])
            if rolling_5_hl != 0:
                volatility_fractality = rolling_2_hl / rolling_5_hl
            else:
                volatility_fractality = 1
        else:
            volatility_fractality = 1
            
        if i >= 4:
            rolling_5_volume = np.mean([data.iloc[j]['volume'] for j in range(i-4, i+1)])
            if rolling_5_volume != 0:
                price_volume_coherence = (current['close'] / current['open']) * (current['volume'] / rolling_5_volume)
            else:
                price_volume_coherence = 1
        else:
            price_volume_coherence = 1
            
        if i >= 1 and data.iloc[i-1]['close'] != 0 and data.iloc[i-1]['volume'] != 0:
            trend_strength = (current['close'] / data.iloc[i-1]['close']) * (current['volume'] / data.iloc[i-1]['volume'])
        else:
            trend_strength = 1
            
        factor4 = volatility_fractality * price_volume_coherence * trend_strength
        
        # Intraday Extreme Reversal with Volume-Price Alignment
        if current['high'] != 0 and current['low'] != 0:
            combined_reversal = (1 - (current['close'] / current['high'])) * ((current['close'] / current['low']) - 1)
        else:
            combined_reversal = 0
            
        if i >= 1 and data.iloc[i-1]['close'] != 0 and data.iloc[i-1]['volume'] != 0:
            volume_price_momentum = (current['close'] / data.iloc[i-1]['close']) * (current['volume'] / data.iloc[i-1]['volume'])
        else:
            volume_price_momentum = 1
            
        if (current['high'] + current['low']) != 0:
            position_context = current['close'] / ((current['high'] + current['low']) / 2)
        else:
            position_context = 1
            
        factor5 = combined_reversal * volume_price_momentum * position_context
        
        # Combine all factors (equal weighting)
        factor.iloc[i] = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    return factor
