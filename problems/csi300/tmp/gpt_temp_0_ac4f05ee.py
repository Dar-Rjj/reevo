import pandas as pd
import pandas as pd
from scipy.stats import pearsonr

def heuristics_v2(df):
    # Measure Short-Term Trend
    short_term_change = df['close'] - df['close'].shift(4)
    rolling_max = df['high'].rolling(window=5, min_periods=1).max()
    rolling_min = df['low'].rolling(window=5, min_periods=1).min()
    price_range = rolling_max - rolling_min
    normalized_short_term_change = short_term_change / price_range
    
    # Measure Long-Term Trend Confirmation
    long_term_change = df['close'] - df['close'].shift(19)
    
    # Roll 20-day Short-Term Trends and Long-Term Changes
    rolling_short_term_trends = normalized_short_term_change.rolling(window=20, min_periods=1).apply(lambda x: pearsonr(x.index, x)[0], raw=False)
    rolling_long_term_changes = long_term_change.rolling(window=20, min_periods=1).apply(lambda x: pearsonr(x.index, x)[0], raw=False)
    
    # Correlation with Short-Term Trend
    trend_confirmation = rolling_short_term_trends.rolling(window=20, min_periods=1).apply(lambda x: pearsonr(x.index, x)[0], raw=False)
    
    return trend_confirmation
