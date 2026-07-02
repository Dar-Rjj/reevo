import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Price Trend Component
    data['Price_Slope_5d'] = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0])
    data['Price_Slope_20d'] = data['close'].rolling(window=20).apply(lambda x: np.polyfit(range(20), x, 1)[0])
    
    # Volume Trend Component
    data['Volume_Slope_5d'] = data['volume'].ewm(span=5).mean().diff().rolling(window=5).mean()
    data['Volume_Slope_20d'] = data['volume'].ewm(span=20).mean().diff().rolling(window=20).mean()
    
    # Divergence Signal
    data['Signal'] = np.where(
        (data['Price_Slope_5d'] > 0) & (data['Volume_Slope_5d'] < 0), -1,
        np.where(
            (data['Price_Slope_5d'] < 0) & (data['Volume_Slope_5d'] > 0), 1, 0
        )
    )
    
    # Normalized Divergence
    data['Price_Volatility_20d'] = data['close'].rolling(window=20).std()
    data['Volume_ZScore_20d'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    
    data['Divergence_Factor'] = data['Signal'] * data['Volume_ZScore_20d'] / data['Price_Volatility_20d']
    
    return data['Divergence_Factor']
