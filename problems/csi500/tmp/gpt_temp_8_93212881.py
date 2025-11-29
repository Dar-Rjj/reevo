import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required technical indicators
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['close_to_open'] = (data['close'] - data['open']) / data['open']
    
    # Volume and liquidity metrics
    data['volume_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma']
    data['amount_ma'] = data['amount'].rolling(window=5).mean()
    data['liquidity_momentum'] = data['amount'] / data['amount_ma']
    
    # Volatility metrics
    data['daily_volatility'] = data['intraday_range'].rolling(window=10).std()
    data['volatility_regime'] = data['daily_volatility'].rolling(window=5).mean()
    data['volatility_change'] = data['volatility_regime'].pct_change()
    
    # Calculate factors for each day
    for i in range(10, len(data)):
        current_data = data.iloc[:i+1].copy()
        current_day = current_data.iloc[-1]
        
        # Factor 1: Intraday Gap Fracture with Liquidity Momentum
        if i >= 15:
            recent_gaps = current_data['gap'].iloc[-5:]
            recent_close_open = current_data['close_to_open'].iloc[-5:]
            
            # Gap fracture detection
            gap_fracture = 0
            for j in range(1, 5):
                if (abs(recent_gaps.iloc[-j]) > 0.01 and 
                    abs(recent_close_open.iloc[-j]) > abs(recent_gaps.iloc[-j]) * 0.8):
                    gap_fracture += 1
            
            # Liquidity momentum
            liquidity_strength = current_data['liquidity_momentum'].iloc[-5:].mean()
            
            # Alignment
            fracture_alignment = gap_fracture / 4.0 if gap_fracture > 0 else 0
            factor1 = gap_fracture * liquidity_strength * fracture_alignment
        else:
            factor1 = 0
        
        # Factor 2: Session-Boundary Reversal with Volume Confirmation
        if i >= 8:
            boundary_gaps = current_data['gap'].iloc[-3:]
            volume_confirms = current_data['volume_momentum'].iloc[-3:]
            
            gap_magnitude = abs(boundary_gaps).mean()
            confirmation_strength = volume_confirms.mean()
            
            # Reversal alignment (negative correlation between gap and next day return)
            if len(boundary_gaps) >= 3:
                reversal_alignment = -np.corrcoef(boundary_gaps.iloc[:-1], 
                                                current_data['close_to_open'].iloc[-3:-1])[0,1]
                reversal_alignment = max(0, reversal_alignment)  # Only positive alignment
            else:
                reversal_alignment = 0
            
            factor2 = gap_magnitude * confirmation_strength * reversal_alignment
        else:
            factor2 = 0
        
        # Factor 3: Volatility Regime Gap Transition
        if i >= 15:
            volatility_shifts = current_data['volatility_change'].iloc[-5:]
            gap_behavior = current_data['gap'].iloc[-5:]
            
            shift_magnitude = abs(volatility_shifts).mean()
            
            # Transition efficiency (gap behavior changes during volatility shifts)
            transition_efficiency = 0
            for j in range(1, 5):
                if abs(volatility_shifts.iloc[-j]) > 0.05:
                    if abs(gap_behavior.iloc[-j]) > abs(gap_behavior.iloc[-j-1]):
                        transition_efficiency += 1
            
            transition_efficiency = transition_efficiency / 4.0
            
            # Alignment precision
            if len(volatility_shifts) >= 3:
                regime_alignment = abs(np.corrcoef(volatility_shifts.iloc[:-1], 
                                                 gap_behavior.iloc[1:])[0,1])
            else:
                regime_alignment = 0
            
            factor3 = shift_magnitude * transition_efficiency * regime_alignment
        else:
            factor3 = 0
        
        # Factor 4: Range Breakout Gap with Liquidity Confirmation
        if i >= 12:
            range_breakouts = []
            for j in range(1, 6):
                if current_data['intraday_range'].iloc[-j] > current_data['intraday_range'].iloc[-j-5:-j].mean() * 1.2:
                    range_breakouts.append(1)
                else:
                    range_breakouts.append(0)
            
            breakout_strength = sum(range_breakouts) / 5.0
            
            # Liquidity confirmation
            liquidity_changes = current_data['liquidity_momentum'].iloc[-5:]
            confirmation_strength = liquidity_changes.mean()
            
            # Breakout-gap alignment
            gap_behavior = current_data['gap'].iloc[-5:]
            if len(range_breakouts) >= 3:
                breakout_alignment = abs(np.corrcoef(range_breakouts[:-1], 
                                                   gap_behavior.iloc[1:])[0,1])
            else:
                breakout_alignment = 0
            
            factor4 = breakout_strength * confirmation_strength * breakout_alignment
        else:
            factor4 = 0
        
        # Combine factors with equal weighting
        total_factors = sum([f != 0 for f in [factor1, factor2, factor3, factor4]])
        if total_factors > 0:
            combined_factor = (factor1 + factor2 + factor3 + factor4) / total_factors
        else:
            combined_factor = 0
        
        factor.iloc[i] = combined_factor
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
