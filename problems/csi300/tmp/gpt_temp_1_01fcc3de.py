import pandas as pd
import numpy as np
def heuristics_v2(data):
    """
    Calculate Price-Volume Divergence Factor based on the divergence between price and volume trends.
    
    Parameters:
    data (pd.DataFrame): DataFrame with columns ['close', 'volume'] and datetime index
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import linregress
    
    close = data['close']
    volume = data['volume']
    
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling linear regression slopes for price and volume
    window = 5
    price_slopes = pd.Series(index=data.index, dtype=float)
    volume_slopes = pd.Series(index=data.index, dtype=float)
    
    for i in range(window-1, len(data)):
        # Price slope calculation
        x = np.arange(window)
        y_price = close.iloc[i-window+1:i+1].values
        slope_price = linregress(x, y_price)[0]
        price_slopes.iloc[i] = slope_price
        
        # Volume slope calculation
        y_volume = volume.iloc[i-window+1:i+1].values
        slope_volume = linregress(x, y_volume)[0]
        volume_slopes.iloc[i] = slope_volume
        
        # Calculate divergence factor
        if (slope_price > 0 and slope_volume < 0) or (slope_price < 0 and slope_volume > 0):
            # Multiply by absolute values to weight the strength of divergence
            divergence = slope_price * slope_volume * -1  # -1 to make divergences positive
            strength = (abs(slope_price) + abs(slope_volume)) / 2
            factor.iloc[i] = divergence * strength * 100
        else:
            factor.iloc[i] = 0
    
    return factor
