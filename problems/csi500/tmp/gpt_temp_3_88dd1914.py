import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-Term Price Reversal Signal
    # Recent Price Momentum
    data['return_2d'] = data['close'].pct_change(2)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    
    # Reversal Threshold
    data['momentum_decel'] = data['return_2d'] - data['return_2d'].shift(1)
    data['price_flip'] = ((data['return_2d'] > 0) & (data['return_2d'].shift(1) < 0)).astype(int) - \
                        ((data['return_2d'] < 0) & (data['return_2d'].shift(1) > 0)).astype(int)
    
    # Liquidity Acceleration Detection
    # Volume Velocity
    data['volume_change'] = data['volume'].pct_change()
    data['volume_trend_3d'] = data['volume'].rolling(window=3).mean() / data['volume'].rolling(window=5).mean() - 1
    
    # Amount Acceleration
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['amount_vs_avg'] = data['amount'] / data['amount_5d_avg'] - 1
    data['amount_growth'] = data['amount'].pct_change(3)
    
    # Cross-Sectional Relative Strength
    # Daily return percentile (cross-sectional)
    data['daily_return'] = data['close'].pct_change()
    data['return_percentile'] = data.groupby(data.index)['daily_return'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Intraday Strength Score
    data['intraday_strength'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Liquidity Momentum Ranking
    # Volume Velocity Percentile (cross-sectional)
    data['volume_velocity'] = data['volume_change'].rolling(window=3).mean()
    data['volume_velocity_pct'] = data.groupby(data.index)['volume_velocity'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Amount Acceleration Rank (cross-sectional)
    data['amount_accel_rank'] = data.groupby(data.index)['amount_vs_avg'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Combine components into final factor
    # Price reversal component (negative for mean reversion)
    price_reversal = -data['return_2d'] * (1 + data['price_flip'])
    
    # Liquidity acceleration component
    liquidity_accel = data['volume_trend_3d'] * data['amount_vs_avg']
    
    # Relative strength component
    relative_strength = data['return_percentile'] * data['intraday_strength'].fillna(0.5)
    
    # Liquidity momentum component
    liquidity_momentum = data['volume_velocity_pct'] * data['amount_accel_rank']
    
    # Final factor combining all components
    factor = (price_reversal * 0.4 + 
              liquidity_accel * 0.3 + 
              relative_strength * 0.2 + 
              liquidity_momentum * 0.1)
    
    return factor
