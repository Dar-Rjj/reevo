import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Price Trend Measurement
    df['short_term_price_slope'] = df['close'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0], raw=True)
    df['medium_term_price_slope'] = df['close'].rolling(window=20).apply(lambda x: np.polyfit(np.arange(20), x, 1)[0], raw=True)
    
    # Volume Divergence
    df['volume_slope'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0], raw=True)
    df['price_volume_divergence'] = np.log(df['volume_slope'] / df['short_term_price_slope'])
    
    # Signal Generation
    df['bullish_confirmation'] = (df['short_term_price_slope'] > 0) & (df['price_volume_divergence'] > 1.0)
    df['bearish_confirmation'] = (df['short_term_price_slope'] < 0) & (df['price_volume_divergence'] < -1.0)
    
    # Final Signal
    df['signal'] = np.where(df['bullish_confirmation'], 1, np.where(df['bearish_confirmation'], -1, 0))
    
    return df['signal']
