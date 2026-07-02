import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Price Compression Signal
    # Measure Price Tightness
    df['price_range'] = (df['high'] - df['low']) / df['open']
    df['price_compression'] = df['price_range'].rolling(5).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    # Directional Bias
    df['direction'] = np.sign(df['close'] - df['open'])
    df['compression_signal'] = df['price_compression'] * df['direction']
    
    # Volume Expansion Signal
    # Volume Spike Detection
    df['vol_median'] = df['volume'].rolling(10).median()
    df['vol_spike'] = np.log1p(df['volume'] / df['vol_median'])
    
    # Volume Trend Confirmation
    df['vol_ma'] = df['volume'].rolling(5).mean()
    df['vol_trend'] = (df['volume'] > df['vol_ma']).astype(int)
    df['volume_signal'] = df['vol_spike'] * df['vol_trend']
    
    # Combined Divergence
    df['combined_divergence'] = df['compression_signal'] * df['volume_signal']
    
    # Scale by Absolute Price Change
    df['abs_price_change'] = abs(df['close'] - df['open']) / df['open']
    df['scaled_divergence'] = df['combined_divergence'] * df['abs_price_change']
    
    # 3-day rolling normalization
    df['factor'] = df['scaled_divergence'].rolling(3).apply(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0, raw=False
    )
    
    return df['factor']
