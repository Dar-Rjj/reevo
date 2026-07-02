import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Measurement
    df['short_term_trend'] = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
    df['medium_term_trend'] = df['close'].rolling(window=20).apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
    
    # Volume Divergence
    df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x).slope, raw=True)
    df['volume_divergence'] = np.log((df['volume_trend'] / df['short_term_trend']).replace([np.inf, -np.inf], np.nan))
    
    # Signal Generation
    bullish_signal = (df['short_term_trend'] > 0) & (df['volume_divergence'] > 0.5)
    bearish_signal = (df['short_term_trend'] < 0) & (df['volume_divergence'] < -0.5)
    
    # Factor Values
    factor_values = pd.Series(0, index=df.index)
    factor_values[bullish_signal] = 1
    factor_values[bearish_signal] = -1
    
    return factor_values
