import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate Price Momentum Component
    def calculate_price_momentum(group):
        return group['close'].rolling(window=5).apply(lambda x: linregress(range(5), x).slope, raw=True)
    
    price_momentum = calculate_price_momentum(data)
    
    # Calculate Volume Momentum Component
    def calculate_volume_momentum(group):
        return group['volume'].rolling(window=5).apply(lambda x: linregress(range(5), x).slope, raw=True)
    
    volume_momentum = calculate_volume_momentum(data)
    
    # Calculate Signal Generation
    signal = price_momentum - volume_momentum
    close_open_range = data['close'] - data['open']
    factor = signal / close_open_range
    
    return factor
