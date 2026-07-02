import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Price Reversal Signal components
    data['intraday_range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    data['open_vs_prev_close'] = data['open'] - data['prev_close']
    
    # Volume Confirmation components
    data['volume_ma5'] = data['volume'].rolling(5).mean()
    data['volume_change'] = data['volume'] - data['volume_ma5']
    data['price_std20'] = data['close'].rolling(20).std()
    
    # Avoid lookahead bias by ensuring all calculations use only past data
    for t in range(1, len(data)):
        current_data = data.iloc[:t+1].copy()
        
        # Calculate Price Reversal Signal
        if current_data['open_vs_prev_close'].iloc[-1] > 0:
            price_signal = -current_data['intraday_range'].iloc[-1] / current_data['prev_close'].iloc[-1]
        else:
            price_signal = current_data['intraday_range'].iloc[-1] / current_data['prev_close'].iloc[-1]
        
        # Calculate Volume Confirmation
        if current_data['price_std20'].iloc[-1] > 0:
            volume_confirmation = current_data['volume_change'].iloc[-1] / current_data['price_std20'].iloc[-1]
        else:
            volume_confirmation = 0
        
        # Combine signals
        factor.iloc[t] = price_signal * volume_confirmation
    
    return factor
