import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Sentiment-Return Divergence
    # EMA(high - low, 5)
    ema_high_low = df['high'] - df['low']
    ema_high_low = ema_high_low.ewm(span=5, adjust=False).mean()
    
    # rolling_return
    rolling_return = df['close'].pct_change().rolling(window=10, min_periods=1).mean()
    
    # rolling_correlation
    rolling_corr = ema_high_low.rolling(window=10, min_periods=1).corr(rolling_return)
    
    # cross_sectional_rank
    normalized_srd = rolling_corr.rank(pct=True)
    
    # Volume Confirmation
    # zscore(amount_t, rolling_mean(amount, 10))
    rolling_mean_amount = df['amount'].rolling(window=10, min_periods=1).mean()
    rolling_std_amount = df['amount'].rolling(window=10, min_periods=1).std()
    zscore_amount = (df['amount'] - rolling_mean_amount) / rolling_std_amount
    
    # rolling_rank(volume, window=10)
    rolling_rank_volume = df['volume'].rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volume Spike Indicator
    volume_spike_indicator = zscore_amount * rolling_rank_volume
    
    # EMA(0.2, 5)
    alpha = 0.2
    ema_decay = alpha * (np.exp(np.arange(5) / 5) - 1) / np.exp(1)
    
    # multiply Sentiment-Return Divergence with Volume Confirmation
    final_factor = normalized_srd * volume_spike_indicator * ema_decay[-1]
    
    return final_factor
