import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic metrics
    df['daily_range'] = df['high'] - df['low']
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_range'] = df['prev_high'] - df['prev_low']
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_amount'] = df['amount'].shift(1)
    
    # Calculate price changes
    df['price_change'] = df['close'] - df['open']
    df['overnight_gap'] = df['open'] - df['prev_close']
    
    # Calculate amount/volume ratio
    df['amount_volume_ratio'] = df['amount'] / df['volume']
    df['prev_amount_volume_ratio'] = df['amount_volume_ratio'].shift(1)
    
    # Calculate volume metrics (assuming first hour = first 1/6.5 of trading day)
    df['first_hour_volume'] = df['volume'] * 0.15  # Approximation
    df['last_hour_volume'] = df['volume'] * 0.15   # Approximation
    df['midday_volume'] = df['volume'] * 0.5       # Approximation
    df['prev_last_hour_volume'] = df['last_hour_volume'].shift(1)
    
    # Calculate price levels (approximations)
    df['midday_price'] = (df['high'] + df['low']) / 2
    df['first_hour_high'] = df['high'] * 0.98      # Approximation
    df['first_hour_low'] = df['low'] * 1.02        # Approximation
    
    # Calculate range metrics
    df['first_hour_range'] = df['first_hour_high'] - df['first_hour_low']
    df['midday_range'] = df['daily_range'] * 0.7   # Approximation
    
    # Calculate directional metrics
    df['close_direction'] = np.sign(df['close'] - df['prev_close'])
    df['volume_direction'] = np.sign(df['volume'] - df['prev_volume'])
    
    # Opening Regime Transition Patterns
    gap_absorption_momentum = (df['overnight_gap'] / (df['daily_range'] + 1e-8)) * df['first_hour_volume']
    opening_range_fracture = (df['daily_range'] / (abs(df['overnight_gap']) + 1e-8)) * (df['volume'] / (df['prev_volume'] + 1e-8))
    morning_liquidity_transfer = ((df['first_hour_volume'] - df['prev_last_hour_volume']) / (df['prev_last_hour_volume'] + 1e-8)) * (df['close'] - df['open'])
    
    # Intraday Liquidity Decoupling Dynamics
    midday_volume_fracture = (df['midday_volume'] / (df['volume'] + 1e-8)) * (df['close'] - df['midday_price']) * df['daily_range']
    liquidity_gap_momentum = (df['midday_range'] / (df['daily_range'] + 1e-8)) * (1 - df['midday_volume'] / (df['volume'] + 1e-8)) * df['price_change']
    trend_fracture_strength = np.sign(df['midday_price'] - df['open']) * (df['close'] - df['midday_price']) * df['amount']
    
    # Closing Session Efficiency Momentum
    end_of_day_expansion = (df['daily_range'] / (df['prev_range'] + 1e-8)) * df['last_hour_volume'] * df['price_change']
    closing_price_discovery = ((df['close'] - df['low']) / (df['daily_range'] + 1e-8)) * (df['last_hour_volume'] / (df['volume'] + 1e-8))
    
    # Calculate consecutive closes (simplified)
    df['close_up'] = (df['close'] > df['prev_close']).astype(int)
    df['consecutive_closes'] = df['close_up'].groupby(df.index).expanding().apply(lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1).reset_index(level=0, drop=True)
    final_hour_persistence = df['consecutive_closes'] * (df['close'] - df['prev_close']) * df['volume']
    
    # Cross-Session Coordination Patterns
    overnight_liquidity_momentum = (df['first_hour_volume'] / (df['prev_last_hour_volume'] + 1e-8)) * df['overnight_gap'] * df['amount']
    session_efficiency = (df['amount_volume_ratio'] - df['prev_amount_volume_ratio']) * df['price_change']
    cross_session_volume_flow = df['close_direction'] * (df['volume'] / (df['prev_volume'] + 1e-8)) * df['daily_range']
    
    # Composite components
    session_transition = gap_absorption_momentum * opening_range_fracture * morning_liquidity_transfer
    intraday_fracture = midday_volume_fracture * liquidity_gap_momentum * trend_fracture_strength
    closing_efficiency = end_of_day_expansion * closing_price_discovery * final_hour_persistence
    
    # Final alpha factor
    alpha_factor = session_transition * intraday_fracture * closing_efficiency * cross_session_volume_flow
    
    # Clean up and return
    result = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
