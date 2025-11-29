import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # High-Low Momentum Divergence
    df['high_momentum'] = df['high'].pct_change()
    df['low_momentum'] = df['low'].pct_change()
    df['momentum_divergence'] = (df['high_momentum'] - df['low_momentum']) / (df['high'] - df['low']).replace(0, np.nan)
    df['divergence_autocorr'] = df['momentum_divergence'].rolling(window=5, min_periods=3).apply(lambda x: x.autocorr(), raw=False)
    df['volume_trend'] = df['volume'].pct_change(periods=3).rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    factor1 = df['momentum_divergence'] * df['divergence_autocorr'] * df['volume_trend'] * df['volume_ratio']
    
    # Opening Gap Range Efficiency
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['gap_range_ratio'] = df['gap'].abs() / df['daily_range'].replace(0, np.nan)
    df['gap_closure'] = (df['close'] - df['open']) / (df['close'].shift(1) - df['open']).replace(0, np.nan)
    df['gap_speed_decay'] = df['gap_closure'].rolling(window=3).mean()
    df['volume_accel'] = df['volume'].pct_change().rolling(window=3).mean()
    factor2 = df['gap_range_ratio'] * df['gap_speed_decay'] * df['volume_accel']
    
    # Price-Volume Trend Divergence
    df['price_trend'] = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    df['volume_trend_slope'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    df['trend_divergence'] = (df['price_trend'] - df['volume_trend_slope']) / (df['price_trend'].abs() + df['volume_trend_slope'].abs()).replace(0, np.nan)
    df['divergence_autocorr2'] = df['trend_divergence'].rolling(window=5, min_periods=3).apply(lambda x: x.autocorr(), raw=False)
    df['dollar_volume'] = df['close'] * df['volume']
    df['dollar_volume_eff'] = df['dollar_volume'] / df['dollar_volume'].rolling(window=20).mean()
    factor3 = df['trend_divergence'] * df['divergence_autocorr2'] * df['dollar_volume_eff']
    
    # Multi-Timeframe Range Momentum
    df['short_range'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close']
    df['long_range'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
    df['range_ratio'] = df['short_range'] / df['long_range'].replace(0, np.nan)
    df['breakout_persistence'] = (df['close'] > df['high'].shift(1)).rolling(window=5).sum()
    df['volume_trend2'] = df['volume'].pct_change(periods=5).rolling(window=5).mean()
    factor4 = df['range_ratio'] * df['breakout_persistence'] * df['volume_trend2']
    
    # Close Location Value Momentum
    df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    df['clv_accel'] = df['clv'].diff().rolling(window=3).mean()
    df['clv_persistence'] = (df['clv'] > df['clv'].shift(1)).rolling(window=5).sum()
    df['volume_accel2'] = df['volume'].pct_change().rolling(window=3).mean()
    factor5 = df['clv_accel'] * df['clv_persistence'] * df['volume_accel2'] * df['daily_range']
    
    # Combine factors with equal weighting
    combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    return combined_factor
