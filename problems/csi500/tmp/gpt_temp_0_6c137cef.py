import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Reversal with Volume-Weighted Price Impact factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # 2. Volume-Weighted Price Impact
    data['price_impact'] = abs(data['close'] - data['open']) / data['amount']
    data['price_impact'] = data['price_impact'].replace([np.inf, -np.inf], np.nan)
    data['volume_weighted_impact'] = data['price_impact'] * data['volume']
    
    # 3. Momentum Reversal Detection
    # 2-day returns for momentum calculation
    data['momentum_2d'] = data['close'].pct_change(periods=2)
    
    # Identify momentum extremes (top and bottom 20%)
    data['momentum_rank'] = data['momentum_2d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Reversal signal: inefficiency during momentum extremes
    data['reversal_signal'] = 0
    extreme_momentum = (data['momentum_rank'] > 0.8) | (data['momentum_rank'] < 0.2)
    data.loc[extreme_momentum, 'reversal_signal'] = data['price_efficiency'] * np.sign(-data['momentum_2d'])
    
    # 4. Volume Confirmation and Timing
    # Volume concentration in key price zones
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['price_range'] = data['high'] - data['low']
    data['volume_concentration'] = data['volume'] / (data['price_range'] + 1e-8)
    
    # Volume-weighted reversal strength
    data['volume_weighted_reversal'] = data['reversal_signal'] * data['volume_concentration']
    
    # Persistence across trading sessions (3-day rolling mean)
    data['reversal_persistence'] = data['volume_weighted_reversal'].rolling(window=3, min_periods=1).mean()
    
    # Final factor: normalized volume-weighted reversal persistence
    factor = data['reversal_persistence'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    
    return factor
