import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Abnormal Volume Signal
    # Rank volume over 20-day lookback window
    volume_rank = data['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Sign of 5-day close price change
    delta_close = data['close'].diff(5)
    sign_delta = np.sign(delta_close)
    
    # Sentiment Confirmation
    # Rolling correlation between returns and volume (5-day)
    returns = data['close'].pct_change()
    rolling_corr = returns.rolling(5).corr(data['volume'])
    
    # Normalized absolute returns
    norm_abs_returns = returns.abs() / returns.abs().rolling(20).mean()
    
    # Combine components
    abnormal_volume = volume_rank * sign_delta
    sentiment_confirmation = rolling_corr * norm_abs_returns
    
    # Final factor combining both signals
    factor = abnormal_volume * sentiment_confirmation
    
    return factor
