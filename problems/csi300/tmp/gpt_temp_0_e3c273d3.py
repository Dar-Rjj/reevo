import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Liquidity Change Signal
    # Calculate rolling delta of volume over a window of 5
    delta_volume = df['volume'].diff().rolling(window=5).sum()
    
    # Rolling rank of delta_volume over a window of 20
    rolling_rank_delta_volume = delta_volume.rolling(window=20).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # Cross-sectional rank of rolling_rank_delta_volume
    cross_sectional_rank = rolling_rank_delta_volume.groupby(df.index.date).rank(pct=True)
    
    # Z-score of cross_sectional_rank
    zscore_cross_sectional_rank = (cross_sectional_rank - cross_sectional_rank.rolling(window=20).mean()) / cross_sectional_rank.rolling(window=20).std()
    
    # Price Momentum Confirmation
    # Calculate EMA of close with a span of 10
    ema_close = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate rolling mean of close over a window of 20
    rolling_mean_close = df['close'].rolling(window=20).mean()
    
    # Ratio of close to rolling_mean_close
    ratio_close_rolling_mean = df['close'] / rolling_mean_close
    
    # Combine the signals
    liquidity_change_signal = zscore_cross_sectional_rank
    price_momentum_confirmation = ratio_close_rolling_mean * ema_close
    
    # Final heuristic factor
    factor = liquidity_change_signal * price_momentum_confirmation
    
    return factor.dropna()
