import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Overnight Gap: Open / Previous Close - 1
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = data['open'] / data['prev_close'] - 1
    
    # Calculate Intraday Reversal: (High + Low) / 2 - Open
    data['intraday_reversal'] = (data['high'] + data['low']) / 2 - data['open']
    
    # Assess Gap Filling: (Close - Open) / (Previous Close - Open)
    denominator = data['prev_close'] - data['open']
    data['gap_filling'] = np.where(
        denominator != 0, 
        (data['close'] - data['open']) / denominator,
        0
    )
    
    # Calculate 3-day Price Momentum: Close / Close from 3 days ago - 1
    data['price_momentum'] = data['close'] / data['close'].shift(3) - 1
    
    # Calculate 5-day Volume-Weighted Momentum: (Close * Volume) / (Close from 5 days ago * Volume from 5 days ago) - 1
    data['vw_momentum'] = (data['close'] * data['volume']) / (data['close'].shift(5) * data['volume'].shift(5)) - 1
    
    # Compute Divergence: Price Momentum - Volume-Weighted Momentum
    data['momentum_divergence'] = data['price_momentum'] - data['vw_momentum']
    
    # Evaluate Momentum Persistence
    data['momentum_direction'] = np.sign(data['price_momentum'])
    data['momentum_streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['momentum_direction'].iloc[i] == data['momentum_direction'].iloc[i-1]:
            streak += 1
        else:
            streak = 1
        data.loc[data.index[i], 'momentum_streak'] = streak
    
    data['momentum_persistence'] = np.sqrt(data['momentum_streak'])
    
    # Calculate Price Impact: (High - Low) / Amount
    data['price_impact'] = (data['high'] - data['low']) / data['amount']
    
    # Compute Volume Concentration: Volume / (Volume from 1 day ago + Volume from 2 days ago + Volume from 3 days ago)
    volume_sum = data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)
    data['volume_concentration'] = data['volume'] / volume_sum
    
    # Calculate Efficiency Score: 1 / (Price Impact * Volume Concentration)
    denominator = data['price_impact'] * data['volume_concentration']
    data['liquidity_efficiency'] = np.where(
        denominator != 0,
        1 / denominator,
        1
    )
    
    # Compute 5-day rolling average of Amount
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=1).mean()
    
    # Calculate liquidity momentum: (Current Amount - 5-day average Amount) / 5-day average Amount
    data['liquidity_momentum'] = (data['amount'] - data['amount_5d_avg']) / data['amount_5d_avg']
    
    # Calculate Relative Range Position: (Close - Low) / (High - Low)
    denominator = data['high'] - data['low']
    data['relative_range'] = np.where(
        denominator != 0,
        (data['close'] - data['low']) / denominator,
        0.5
    )
    
    # Compute Pressure Index: Relative Range Position * Volume
    data['pressure_index'] = data['relative_range'] * data['volume']
    
    # Calculate Average True Range over 3 days
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_3d'] = data['true_range'].rolling(window=3, min_periods=1).mean()
    
    # Synthesize Composite Factor
    # Gap-Reversal Logic: Overnight Gap * Intraday Reversal * Gap Filling
    gap_reversal = data['overnight_gap'] * data['intraday_reversal'] * data['gap_filling']
    
    # Incorporate Momentum Divergence: Multiply by Divergence * Momentum Persistence
    momentum_component = data['momentum_divergence'] * data['momentum_persistence']
    
    # Apply Liquidity Adjustment: Multiply by Liquidity Efficiency * Liquidity Momentum
    liquidity_adjustment = data['liquidity_efficiency'] * data['liquidity_momentum']
    
    # Add Pressure Component: Multiply by Pressure Index
    pressure_component = data['pressure_index']
    
    # Combine all components
    composite_factor = gap_reversal * momentum_component * liquidity_adjustment * pressure_component
    
    # Scale by Volatility Component (ATR)
    final_factor = composite_factor * data['atr_3d']
    
    # Return the factor series
    return final_factor
