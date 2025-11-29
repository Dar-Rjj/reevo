import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price-Volume Divergence Signal
    volume_weighted_trend = (df['close'] - df['open']) * df['volume']
    raw_price_trend = df['close'] - df['open']
    avg_volume_10d = df['volume'].rolling(window=10, min_periods=5).mean()
    divergence_score = volume_weighted_trend - raw_price_trend * avg_volume_10d
    
    # Construct Dynamic Volatility Regime
    intraday_volatility = df['high'] - df['low']
    volatility_regime = intraday_volatility.rolling(window=6, min_periods=3).median()
    volatility_multiplier = intraday_volatility / volatility_regime
    
    # Build Dynamic Weighted Alpha Factor
    volatility_weight = volatility_multiplier.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) * 1.5 + 1.0 if len(x.dropna()) >= 5 else 1.0, 
        raw=False
    )
    volume_confirmation = df['volume'].rolling(window=3, min_periods=2).sum()
    
    # Combine all components
    alpha_factor = divergence_score * volatility_weight * volume_confirmation
    
    return alpha_factor
