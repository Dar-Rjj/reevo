import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Price Reversal Component
    # Midday Price Rejection
    midday_price_rejection = (data['high'] + data['low']) / 2 - data['close']
    
    # Opening Gap Context
    previous_close = data['close'].shift(1)
    opening_gap_context = (data['open'] - previous_close) / previous_close
    
    # Combine Price Reversal Components
    price_reversal = midday_price_rejection * opening_gap_context
    
    # Liquidity Adjustment
    # Volume Surprise
    volume_ma_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_surprise = data['volume'] / volume_ma_10
    
    # Turnover Ratio
    # Assuming 'amount' is the product of 'close' and 'volume'
    data['amount'] = data['close'] * data['volume']
    amount_ma_20 = data['amount'].rolling(window=20, min_periods=1).mean()
    turnover_ratio = data['amount'] / amount_ma_20
    
    # Combine Liquidity Components
    liquidity_adjustment = volume_surprise * turnover_ratio
    
    # Signal Integration
    combined_signal = price_reversal * liquidity_adjustment
    
    # Normalization: Cross-Sectional Z-score
    factor = combined_signal.groupby(combined_signal.index).transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
