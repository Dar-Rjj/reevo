import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Price Acceleration Component
    daily_return = (df['close'] - df['open']) / df['open']
    price_roc = daily_return.rolling(window=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)

    # Volume Acceleration Component
    volume_change = df['volume'] / df['volume'].shift(3)
    volume_change_ma = volume_change.rolling(window=5).mean()

    # Combine Signals
    acceleration_factor = price_roc * volume_change_ma

    return acceleration_factor
