import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['daily_range'] = (df['high'] - df['low']) / df['low']
    df['price_change'] = df['close'] / df['prev_close'] - 1
    
    # High-Low Range Momentum Divergence
    df['range_ma5'] = df['daily_range'].rolling(window=5).mean()
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['range_momentum'] = df['daily_range'] / df['range_ma5'] - 1
    df['range_price_divergence'] = df['range_momentum'] - df['price_momentum']
    
    # Intraday Reversal Strength
    df['opening_gap'] = (df['open'] / df['prev_close'] - 1)
    df['intraday_reversal'] = (df['close'] / df['open'] - 1) * np.sign(-df['opening_gap'])
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['reversal_strength'] = df['intraday_reversal'] * df['volume_ratio']
    
    # Volatility-Adjusted Amount Flow
    df['price_change_sign'] = np.sign(df['close'] - df['prev_close'])
    df['raw_amount_flow'] = df['amount'] * df['price_change_sign']
    df['volatility_ma10'] = df['daily_range'].rolling(window=10).mean()
    df['volatility_adjusted_flow'] = df['raw_amount_flow'] / (df['volatility_ma10'] + 1e-8)
    
    # Multi-Timeframe Price Compression
    df['range_ma20'] = df['daily_range'].rolling(window=20).mean()
    df['compression_ratio'] = df['daily_range'] / df['range_ma20']
    df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    df['compression_signal'] = (1 - df['compression_ratio']) * df['volume_trend']
    
    # Opening Auction Efficiency
    df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['opening_volume_ratio'] = df['volume'].rolling(window=10).apply(lambda x: x.iloc[0] / x.mean() if len(x) == 10 else np.nan)
    df['auction_efficiency'] = df['open_position'] * df['opening_volume_ratio']
    
    # Price-Volume Trend Consistency
    df['price_trend_5'] = df['close'].rolling(window=5).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 5 else np.nan)
    df['volume_trend_5'] = df['volume'].rolling(window=5).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 5 else np.nan)
    df['trend_consistency'] = df['price_trend_5'] * df['volume_trend_5']
    
    # Liquidity Imbalance Signal
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['volume_concentration'] = df['volume'].rolling(window=5).std() / df['volume'].rolling(window=5).mean()
    df['liquidity_imbalance'] = df['close_position'] * df['volume_concentration']
    
    # Intraday Momentum Divergence
    df['intraday_momentum'] = (df['high'] - df['low']) / df['low']
    df['daily_return'] = df['close'] / df['prev_close'] - 1
    df['intraday_divergence'] = df['intraday_momentum'] * np.sign(df['daily_return'])
    
    # Combine all factors with equal weights
    factors = [
        'range_price_divergence',
        'reversal_strength', 
        'volatility_adjusted_flow',
        'compression_signal',
        'auction_efficiency',
        'trend_consistency',
        'liquidity_imbalance',
        'intraday_divergence'
    ]
    
    # Calculate combined factor (z-score weighted average)
    for factor in factors:
        df[f'{factor}_z'] = (df[factor] - df[factor].rolling(window=20).mean()) / (df[factor].rolling(window=20).std() + 1e-8)
    
    z_factors = [f'{factor}_z' for factor in factors]
    df['combined_factor'] = df[z_factors].mean(axis=1)
    
    result = df['combined_factor']
    
    return result
