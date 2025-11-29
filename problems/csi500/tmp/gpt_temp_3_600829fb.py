import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for current_date in df.index:
        current_data = df.loc[:current_date]
        
        if len(current_data) < 5:  # Need sufficient history
            result.loc[current_date] = 0
            continue
            
        # Get current day data
        current = current_data.iloc[-1]
        prev_close = current_data.iloc[-2]['close'] if len(current_data) > 1 else current['open']
        
        # 1. Early Session Price Behavior
        opening_gap = (current['open'] - prev_close) / prev_close
        first_hour_range = (current['high'] - current['low']) / current['open']
        
        # Opening momentum strength (first hour movement from open)
        if current['high'] != current['low']:
            opening_momentum = (current['close'] - current['open']) / (current['high'] - current['low'])
        else:
            opening_momentum = 0
        
        # Gap preservation (how much of opening gap remains)
        if opening_gap != 0:
            gap_preservation = (current['close'] - prev_close) / (current['open'] - prev_close)
        else:
            gap_preservation = 1
        
        # 2. Identify Reversal Candidates
        strong_opening = abs(opening_gap) > 0.005  # 0.5% gap threshold
        weak_follow_through = abs(opening_momentum) < 0.3  # Weak momentum continuation
        
        # Gap fill measure
        if opening_gap > 0:
            gap_fill = (current['high'] - current['open']) / (current['high'] - current['low']) if current['high'] != current['low'] else 0
        else:
            gap_fill = (current['open'] - current['low']) / (current['high'] - current['low']) if current['high'] != current['low'] else 0
        
        # 3. Session Range Utilization
        if current['high'] != current['low']:
            range_utilization = (current['close'] - current['low']) / (current['high'] - current['low'])
        else:
            range_utilization = 0.5
        
        # 4. Momentum Quality Through Session
        # Use recent history to assess typical behavior
        recent_data = current_data.tail(5)
        typical_range = (recent_data['high'] - recent_data['low']).mean() / recent_data['close'].mean()
        
        if typical_range > 0:
            normalized_range = first_hour_range / typical_range
        else:
            normalized_range = 1
        
        # Directional consistency (how close is close to session extreme)
        if opening_gap > 0:
            directional_consistency = (current['close'] - current['low']) / (current['high'] - current['low']) if current['high'] != current['low'] else 0.5
        else:
            directional_consistency = (current['high'] - current['close']) / (current['high'] - current['low']) if current['high'] != current['low'] else 0.5
        
        # 5. Reversal Strength
        opening_vs_close = opening_momentum * (1 if opening_gap > 0 else -1)
        
        # Session extreme retests
        if current['high'] != current['low']:
            high_retest = (current['close'] - current['low']) / (current['high'] - current['low'])
            low_retest = (current['high'] - current['close']) / (current['high'] - current['low'])
        else:
            high_retest = low_retest = 0.5
        
        # 6. Volume Analysis
        if current['volume'] > 0:
            # Volume concentration at extremes
            typical_volume = recent_data['volume'].mean()
            volume_ratio = current['volume'] / typical_volume if typical_volume > 0 else 1
            
            # Amount efficiency (amount per price movement)
            if abs(current['close'] - current['open']) > 0:
                amount_efficiency = current['amount'] / (abs(current['close'] - current['open']) * current['volume']) if current['volume'] > 0 else 0
            else:
                amount_efficiency = 0
        else:
            volume_ratio = 1
            amount_efficiency = 0
        
        # 7. Combine Components
        # Momentum reversal component
        momentum_component = (
            opening_momentum * (1 - gap_preservation) *  # Reversal from opening momentum
            (1 if strong_opening and weak_follow_through else 0.5) *  # Reversal candidate
            gap_fill  # Gap fill progress
        )
        
        # Efficiency component
        efficiency_component = (
            (1 - abs(range_utilization - 0.5)) *  # Balanced range utilization
            directional_consistency *  # Session consistency
            (1 - abs(normalized_range - 1))  # Normalized range
        )
        
        # Volume confirmation component
        volume_component = (
            volume_ratio *  # Volume intensity
            (amount_efficiency if amount_efficiency > 0 else 1)  # Transaction quality
        )
        
        # Final factor calculation
        reversal_efficiency = (
            momentum_component *
            efficiency_component *
            volume_component *
            (1 if opening_gap > 0 else -1)  # Direction based on opening gap
        )
        
        result.loc[current_date] = reversal_efficiency
    
    return result
