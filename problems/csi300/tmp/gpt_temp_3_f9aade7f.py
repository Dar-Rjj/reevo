import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(data, window_spread=20, window_skew=60):
    """
    Calculate rolling asymmetry in intraday prices based on high-low spread skewness.
    
    Parameters:
    - data: DataFrame with columns ['high', 'low', 'close']
    - window_spread: int, window size for high-low spread calculation (default 20)
    - window_skew: int, window size for skewness calculation (default 60)
    
    Returns:
    - pd.Series: factor values indexed by date
    """
    # Calculate daily high-low spread
    daily_spread = data['high'] - data['low']
    
    # Normalize spread by closing price
    normalized_spread = daily_spread / data['close']
    
    # Calculate rolling skewness of normalized spread
    # Using min_periods=1 to ensure we get values even with incomplete windows
    factor = normalized_spread.rolling(
        window=window_skew,
        min_periods=1
    ).apply(lambda x: skew(x.dropna()))
    
    return factor
