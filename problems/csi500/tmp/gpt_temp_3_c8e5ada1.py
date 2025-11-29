import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Divergence Framework
    intraday_return = (df['close'] - df['open']) / df['open']
    five_day_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    price_divergence = intraday_return - five_day_momentum
    
    # Volume Confirmation System
    volume_anomaly = df['volume'] / df['volume'].rolling(20).mean()
    volume_price_corr = df['volume'].rolling(10).corr(df['close'] - df['open'])
    volume_confirmation = volume_anomaly * abs(volume_price_corr)
    
    # Volatility Adjustment
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low_range, np.maximum(high_prev_close, low_prev_close))
    volatility_adjusted_divergence = price_divergence * volume_confirmation / true_range.rolling(5).mean()
    
    # Efficiency Component
    resistance_pressure = (df['high'] - df['close']) / (df['high'] - df['low'])
    support_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'])
    efficiency_score = resistance_pressure * support_efficiency
    
    # Liquidity Integration
    volume_volatility_ratio = df['volume'] / true_range
    efficiency_liquidity = efficiency_score * volume_volatility_ratio
    
    # Trend Persistence
    def linear_slope(x):
        if len(x) < 2:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]
    
    liquidity_trend = df['volume'].rolling(10).apply(linear_slope, raw=False)
    momentum_consistency = np.sign(five_day_momentum) * np.sign(liquidity_trend)
    persistence_weight = momentum_consistency.rolling(5).mean()
    
    # Final Factor
    core_factor = volatility_adjusted_divergence * efficiency_liquidity
    trend_weighted_alpha = core_factor * persistence_weight
    
    return trend_weighted_alpha
