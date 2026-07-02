import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Trend: Rolling 5-day Slope of Close price
    price_slope = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    
    # Calculate Volume Trend: Rolling 5-day Slope of Volume
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    
    # Detect Divergence Direction
    price_up = price_slope > 0
    price_down = price_slope < 0
    volume_up = volume_slope > 0
    volume_down = volume_slope < 0
    
    # Generate Trading Signal
    short_signal = price_up & volume_down  # Price Up & Volume Down → Short
    long_signal = price_down & volume_up   # Price Down & Volume Up → Long
    
    # Combine signals into factor values
    factor = pd.Series(0, index=df.index)
    factor[short_signal] = -1  # Short signal
    factor[long_signal] = 1    # Long signal
    
    return factor
