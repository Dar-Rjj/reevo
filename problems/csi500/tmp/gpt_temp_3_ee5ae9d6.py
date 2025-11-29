import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Gap Ratio
    df = df.copy()
    df['gap_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Identify Extreme Gaps using rolling percentiles (20-day lookback)
    df['gap_rank'] = df['gap_ratio'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, 
        raw=False
    )
    
    # Create extreme gap indicators
    df['extreme_positive_gap'] = (df['gap_rank'] >= 0.9).astype(int)
    df['extreme_negative_gap'] = (df['gap_rank'] <= 0.1).astype(int)
    
    # Calculate next-day returns (shifted forward for reversal detection)
    df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
    
    # Detect Price Reversals
    df['positive_gap_reversal'] = df['extreme_positive_gap'] * (-df['next_day_return'])
    df['negative_gap_reversal'] = df['extreme_negative_gap'] * df['next_day_return']
    
    # Volume Confirmation - calculate rolling average volume (20-day lookback)
    df['avg_volume_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['high_volume'] = (df['volume'] > df['avg_volume_20d']).astype(int)
    
    # Combine reversal signals with volume confirmation
    df['reversal_signal'] = (
        df['positive_gap_reversal'] * df['high_volume'] + 
        df['negative_gap_reversal'] * df['high_volume']
    )
    
    # Create final factor (smoothed with 5-day moving average)
    factor = df['reversal_signal'].rolling(window=5, min_periods=3).mean()
    
    return factor
