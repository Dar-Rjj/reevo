import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day values
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Calculate intraday hourly data (assuming first hour = first 1/6.5 of trading day, last hour = last 1/6.5)
    # For simplicity, we'll use the first and last 15% of daily range as proxy for first and last hour
    data['first_hour_open'] = data['open']
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_close'] = (data['open'] + data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)) / 2
    data['first_hour_amount'] = data['amount'] * 0.15  # Proxy for first hour amount
    
    data['last_hour_open'] = (data['high'] + data['low']) / 2  # Proxy for last hour open
    data['last_hour_amount'] = data['amount'] * 0.15  # Proxy for last hour amount
    
    # Opening Gap Reversal Dynamics
    data['gap_pressure'] = (data['open'] - data['prev_close']) * np.sign(data['open'] - data['prev_close'])
    data['gap_flow_alignment'] = data['gap_pressure'] * np.sign(data['first_hour_amount'] - data['prev_amount'])
    data['opening_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) - (np.minimum(data['open'], data['close']) - data['low'])
    
    # Intraday Flow Compression Patterns
    data['session_flow_concentration'] = (data['first_hour_amount'] - data['last_hour_amount']) / data['amount']
    data['range_compression'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'])
    data['flow_compression'] = data['amount'] / data['prev_amount']
    
    # Reversal Confirmation Signals
    data['early_session_efficiency'] = np.abs(data['first_hour_close'] - data['first_hour_open']) / (data['first_hour_amount'] + 1e-8)
    
    # Failed Breakout Detection
    data['failed_breakout_high'] = ((data['high'] == data['first_hour_high']) & (data['close'] < data['high'])).astype(float)
    data['failed_breakout_low'] = ((data['low'] == data['first_hour_low']) & (data['close'] > data['low'])).astype(float)
    data['failed_breakout_detection'] = data['failed_breakout_high'] + data['failed_breakout_low']
    
    data['flow_momentum_divergence'] = np.sign(data['first_hour_close'] - data['first_hour_open']) * np.sign(data['close'] - data['first_hour_close'])
    
    # Closing Auction Validation
    data['auction_reversal_pressure'] = (data['close'] - data['last_hour_open']) / (data['last_hour_amount'] + 1e-8)
    data['flow_concentration_shift'] = np.sign(data['last_hour_amount'] - data['first_hour_amount'])
    data['closing_efficiency'] = np.abs(data['close'] - data['last_hour_open']) / (data['last_hour_amount'] + 1e-8)
    
    # Final Integration
    data['opening_regime'] = data['gap_flow_alignment'] * data['opening_rejection'] * data['early_session_efficiency']
    data['compression_regime'] = data['session_flow_concentration'] * data['range_compression'] * data['flow_compression']
    data['reversal_regime'] = data['failed_breakout_detection'] * data['flow_momentum_divergence'] * data['auction_reversal_pressure']
    
    # Final Factor Output
    factor = (data['opening_regime'] * data['compression_regime'] * data['reversal_regime'] * 
              data['flow_concentration_shift'] * data['closing_efficiency'])
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
