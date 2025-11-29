import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily compression ratio
    data['compression_ratio'] = (data['high'] - data['low']) / (data['close'] - data['open']).replace(0, np.nan)
    data['compression_ratio'] = data['compression_ratio'] * data['close']
    
    # Calculate 10-day rolling compression percentile
    data['compression_percentile'] = data['compression_ratio'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 5 else np.nan
    )
    
    # Classify compression regimes (High compression = top 30th percentile)
    data['high_compression'] = (data['compression_percentile'] >= 0.7).astype(float)
    
    # Calculate volume expansion signal
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_expansion'] = data['volume'] / data['volume_median_10d']
    data['volume_compression_signal'] = data['volume_expansion'] * data['high_compression']
    
    # Generate breakout probability factor
    data['breakout_factor'] = np.where(
        (data['high_compression'] == 1) & (data['volume_expansion'] > 1),
        data['volume_compression_signal'],
        0
    )
    
    # Calculate gap absorption
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']).abs().replace(0, np.nan)
    data['gap_absorption'] = (data['close'] - data['open']) / data['overnight_gap']
    data['gap_absorption'] = data['gap_absorption'].clip(-2, 2)  # Bound extreme values
    
    # Multiply breakout factor by gap absorption strength
    data['composite_factor'] = data['breakout_factor'] * data['gap_absorption']
    
    # Compute 20-day signal persistence measure (absolute value rolling mean)
    data['signal_persistence'] = data['composite_factor'].abs().rolling(window=20, min_periods=10).mean()
    
    # Weight composite factor by persistence stability
    data['persistence_weight'] = 1 / (1 + data['signal_persistence'].replace(0, np.nan))
    data['final_factor'] = data['composite_factor'] * data['persistence_weight']
    
    return data['final_factor']
