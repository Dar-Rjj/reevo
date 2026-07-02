import pandas as pd
import pandas as pd
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Slope using Close price over 5 days
    price_slopes = df['close'].rolling(window=5).apply(lambda x: linregress(range(5), x).slope, raw=True)
    
    # Calculate Volume Slope using Volume over 5 days
    volume_slopes = df['volume'].rolling(window=5).apply(lambda x: linregress(range(5), x).slope, raw=True)
    
    # Compute Divergence by subtracting Volume Slope from Price Slope and multiplying by Current Volume
    divergence = (price_slopes - volume_slopes) * df['volume']
    
    # Return the divergence as the factor
    return divergence
