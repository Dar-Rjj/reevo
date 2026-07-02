import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Measure Price Efficiency
    intraday_return = df['close'] / df['open']
    historical_efficiency = df['close'].rolling(window=5).apply(lambda x: x.iloc[:-1].mean()) / df['open'].rolling(window=5).apply(lambda x: x.iloc[:-1].mean())
    price_efficiency_divergence = intraday_return - historical_efficiency
    
    # Measure Volume Divergence
    volume_spike = df['volume'] / df['volume'].rolling(window=10).mean()
    volume_divergence = np.log(volume_spike).abs()
    
    # Combine Components
    combined_factor = price_efficiency_divergence * volume_divergence
    standardized_factor = (combined_factor - combined_factor.mean()) / combined_factor.std()
    
    return standardized_factor
