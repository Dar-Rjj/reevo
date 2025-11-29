import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Compression Analysis
    # True Range Calculation
    df['prev_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = np.abs(df['high'] - df['prev_close'])
    df['low_prev_close'] = np.abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # Compression Duration Tracking
    df['compression_flag'] = (df['true_range'] < df['true_range'].rolling(window=10).mean()).astype(int)
    df['compression_duration'] = df['compression_flag'].rolling(window=20, min_periods=1).apply(
        lambda x: len(x) - np.argmax(x[::-1]) if np.any(x) else 0, raw=False
    )
    
    # Breakout Signal Detection
    df['price_range_ratio'] = (df['high'] - df['low']) / df['close']
    df['breakout_signal'] = ((df['price_range_ratio'] > df['price_range_ratio'].rolling(window=10).mean() * 1.5) & 
                            (df['compression_duration'] >= 3)).astype(int)
    
    # Momentum Persistence Assessment
    # Session Momentum Strength
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['momentum_strength'] = df['intraday_return'].rolling(window=5).apply(
        lambda x: np.sum(x[x > 0]) / (np.sum(np.abs(x)) + 1e-8), raw=False
    )
    
    # Momentum Transfer Efficiency
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['momentum_efficiency'] = df['intraday_return'] / (np.abs(df['overnight_gap']) + 1e-8)
    df['momentum_efficiency'] = df['momentum_efficiency'].replace([np.inf, -np.inf], 0)
    
    # Volume-Efficiency Integration
    # Volume-Weighted Price Impact
    df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    df['price_vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    df['volume_price_impact'] = df['price_vwap_deviation'] * np.log1p(df['volume'])
    
    # Volume Dynamics During Compression
    df['volume_compression_ratio'] = (df['volume'].rolling(window=10).mean() / 
                                     df['volume'].rolling(window=20).mean())
    df['volume_efficiency'] = df['volume_price_impact'] * (1 - df['volume_compression_ratio'])
    
    # Composite Signal Generation
    # Momentum-Confirmed Breakouts
    df['momentum_confirmed'] = (df['breakout_signal'] * 
                               (df['momentum_strength'] > 0.6) * 
                               (df['momentum_efficiency'] > 0))
    
    # Volume-Efficiency Filtered Signals
    df['volume_filtered'] = (df['momentum_confirmed'] * 
                            (df['volume_efficiency'] > df['volume_efficiency'].rolling(window=10).quantile(0.3)))
    
    # Final composite factor
    factor = (df['volume_filtered'] * 
             df['momentum_strength'] * 
             df['momentum_efficiency'] * 
             df['volume_efficiency'])
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'high_low', 'high_prev_close', 'low_prev_close', 
                   'true_range', 'compression_flag', 'compression_duration', 
                   'price_range_ratio', 'breakout_signal', 'intraday_return', 
                   'momentum_strength', 'overnight_gap', 'momentum_efficiency',
                   'vwap', 'price_vwap_deviation', 'volume_price_impact',
                   'volume_compression_ratio', 'volume_efficiency',
                   'momentum_confirmed', 'volume_filtered']
    
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return factor
