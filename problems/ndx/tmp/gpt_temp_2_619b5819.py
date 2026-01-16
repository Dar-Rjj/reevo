import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Price Momentum
    # Short-Term Momentum (10-day)
    momentum_10d = df['close'] / df['close'].shift(10)
    
    # Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['close']
    
    # Volume Momentum
    # Volume Change (10-day)
    volume_change = df['volume'] / df['volume'].shift(10)
    
    # Volume Confirmation (10-day rolling mean)
    rolling_volume_mean = df['volume'].rolling(window=10).mean()
    volume_confirmation = df['volume'] / rolling_volume_mean
    
    # Combine signals
    combined_factor = (momentum_10d * intraday_momentum * 
                       volume_change * volume_confirmation)
    
    # Normalize cross-sectionally
    def normalize(series):
        mean = series.mean()
        std = series.std()
        if std == 0:  # avoid division by zero
            return pd.Series(0, index=series.index)
        normalized = (series - mean) / std
        return normalized.clip(-4, 4)
    
    # Apply normalization daily
    normalized_factor = combined_factor.groupby(combined_factor.index).transform(normalize)
    
    return normalized_factor
