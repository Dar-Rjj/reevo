import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Rejection Momentum
    momentum_reversal = np.sign((df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1)) * ((df['close'] - df['open']) / df['open'])
    net_rejection = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    rejection_momentum = momentum_reversal * net_rejection
    
    # Liquidity-Weighted Efficiency
    price_range_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    volume_flow = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
    liquidity_multiplier = np.log(df['amount']) / np.log(df['amount'].rolling(5).mean())
    liquidity_weighted_efficiency = price_range_efficiency * volume_flow * liquidity_multiplier
    
    # Volatility Adjustment
    volatility_ratio = ((df['high'] - df['low']) / df['open']) / ((df['high'] - df['low']).rolling(5).mean() / df['open'])
    volatility_adjustment = 1 / (1 + volatility_ratio)
    
    # Accumulation Pattern
    price_change = df['close'] - df['close'].shift(1)
    accumulation_strength = price_change.rolling(3).apply(lambda x: (x * df.loc[x.index, 'volume']).sum(), raw=False) / (np.abs(price_change.rolling(3).apply(lambda x: (x * df.loc[x.index, 'volume']).sum(), raw=False)) + 1e-8)
    
    # Composite Factor
    core_component = rejection_momentum * liquidity_weighted_efficiency * volatility_adjustment
    accumulation_multiplier = np.where(accumulation_strength > 0.8, 1.4, np.where(accumulation_strength < -0.8, 0.6, 1.0))
    final_factor = core_component * accumulation_multiplier
    
    return final_factor
