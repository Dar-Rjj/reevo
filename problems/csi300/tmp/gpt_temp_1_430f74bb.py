import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Intraday Price Deviation
    intraday_price_deviation = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Historical Consistency (5-day STD of Intraday Price Deviation)
    historical_consistency = intraday_price_deviation.rolling(window=5).std()
    
    # Volume Concentration
    volume_concentration = data['volume'] / (data['high'] - data['low'])
    
    # Normalization of Volume Concentration (Z-Score)
    volume_concentration_zscore = volume_concentration.groupby(data.index.date).transform(zscore)
    
    # Price Deviation Measurement
    price_deviation_measurement = intraday_price_deviation * historical_consistency
    
    # Volume Efficiency
    volume_efficiency = volume_concentration_zscore
    
    # Factor Construction (Combine Components)
    factor = price_deviation_measurement * volume_efficiency
    
    # Normalization (Cross-sectional Rank)
    factor_rank = factor.groupby(data.index.date).rank(pct=True)
    
    return factor_rank
