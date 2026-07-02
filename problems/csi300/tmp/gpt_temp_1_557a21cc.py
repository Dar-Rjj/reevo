import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate intraday volatility (normalized range)
    intraday_range = data['high'] - data['low']
    normalized_range = (intraday_range / data['close'].shift(1)) * 100
    
    # Calculate momentum factors
    short_term_momentum = data['close'].rolling(3).mean()
    medium_term_momentum = data['close'].rolling(10).mean()
    
    # Calculate momentum divergence
    momentum_divergence = short_term_momentum - medium_term_momentum
    mad = momentum_divergence.rolling(20).apply(lambda x: np.median(np.abs(x - np.median(x))))
    normalized_divergence = momentum_divergence / (mad + 1e-6)  # Avoid division by zero
    
    # Apply logistic scaling to divergence
    scaled_divergence = 1 / (1 + np.exp(-normalized_divergence)) - 0.5
    
    # Combine with intraday volatility
    factor = scaled_divergence * normalized_range
    
    return factor
