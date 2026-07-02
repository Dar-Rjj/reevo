import numpy as np
def heuristics_v2(df):
    # Price Inefficiency component
    # Calculate absolute ratio of open to rolling mean of high/low
    rolling_mean_high_low = (df['high'].rolling(window=10, min_periods=5).mean() + 
                             df['low'].rolling(window=10, min_periods=5).mean()) / 2
    abs_ratio = (df['open'] / rolling_mean_high_low).abs()
    
    # Normalize using rolling rank (cross-sectional)
    def rolling_rank(series):
        return series.rolling(window=15, min_periods=5).apply(
            lambda x: (x[-1] > x[:-1]).mean() if len(x[:-1]) > 0 else np.nan
        )
    
    price_inefficiency = abs_ratio.groupby(level=0).apply(rolling_rank)
    
    # Liquidity Gap component
    # Difference between current volume and rolling median
    volume_median_diff = df['volume'] - df['volume'].rolling(window=5, min_periods=3).median()
    
    # Correlate with price inefficiency and rolling std of close
    rolling_std_close = df['close'].rolling(window=10, min_periods=5).std()
    liquidity_gap = volume_median_diff * price_inefficiency * rolling_std_close
    
    # Combine components to get final anomaly score
    microstructure_anomaly = price_inefficiency + liquidity_gap
    
    return microstructure_anomaly
