import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Divergence Component
    # 3-day momentum
    mom_3d = df['close'] / df['close'].shift(3) - 1
    # 8-day momentum
    mom_8d = df['close'] / df['close'].shift(8) - 1
    # Divergence calculation
    momentum_divergence = (mom_3d - mom_8d) / (np.abs(mom_8d) + 1e-8)
    # Intraday volatility scaling
    intraday_vol = (df['high'] - df['low']) / df['close']
    momentum_component = momentum_divergence * intraday_vol
    
    # Volume-Price Efficiency Component
    # Price move efficiency
    abs_price_change = np.abs(df['close'] - df['open'])
    efficiency_ratio = abs_price_change / (df['high'] - df['low'] + 1e-8)
    # Volume intensity adjustment
    volume_ratio = df['volume'] / df['volume'].rolling(15).mean()
    # Directional confirmation
    directional_bias = np.sign(df['close'] - df['open'])
    volume_component = efficiency_ratio * volume_ratio * directional_bias
    
    # Trend Persistence Component
    # Trend consistency
    trend_5d = np.sign(df['close'] - df['close'].shift(5))
    trend_2d = np.sign(df['close'] - df['close'].shift(2))
    persistence = (trend_5d + trend_2d) / 2
    # Volatility normalization
    vol_normalization = df['close'].rolling(12).std() / df['close']
    trend_component = persistence * vol_normalization
    
    # Factor Combination
    raw_factor = momentum_component * volume_component * trend_component
    
    # Cross-sectional z-score transformation
    factor = raw_factor.groupby(raw_factor.index).transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
