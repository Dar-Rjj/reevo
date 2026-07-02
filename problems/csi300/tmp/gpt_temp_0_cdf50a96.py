import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Short-Term Reversal Signal
    # Calculate daily price change
    df['delta_close'] = df['close'] - df['close'].shift(1)
    
    # Calculate rolling rank of price changes over 5 days
    df['reversal_raw'] = df['delta_close'].rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Liquidity Confirmation
    # Calculate volume ratio: current volume vs EMA(volume, span=10)
    df['volume_ema'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ema']
    
    # Calculate z-score of volume changes using rolling 20-day std
    df['delta_volume'] = df['volume'].diff()
    rolling_std = df['delta_volume'].rolling(20).std()
    df['volume_zscore'] = df['delta_volume'] / rolling_std
    
    # Combine signals with equal weights
    factor = 0.5 * df['reversal_raw'] + 0.5 * df['volume_zscore']
    
    return factor.dropna()
