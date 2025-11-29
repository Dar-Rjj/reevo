import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate basic price metrics
    data['midday_price'] = (data['high'] + data['low']) / 2
    data['daily_range'] = data['high'] - data['low']
    
    # Morning Session Momentum Strength
    data['morning_momentum'] = (data['midday_price'] - data['open']) / data['open']
    data['morning_momentum_abs'] = abs(data['morning_momentum'])
    
    # Afternoon Session Reversal Quality
    data['afternoon_reversal'] = (data['close'] - data['midday_price']) / data['midday_price']
    data['reversal_efficiency'] = abs(data['afternoon_reversal']) / (data['daily_range'] / data['open'] + 1e-8)
    
    # Momentum Overextension Signals
    data['price_range_deviation'] = (data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    data['extreme_move'] = ((data['morning_momentum_abs'] > data['morning_momentum_abs'].rolling(5).mean()) & 
                           (abs(data['afternoon_reversal']) > abs(data['afternoon_reversal']).rolling(5).mean())).astype(int)
    
    # Volume Analysis
    data['am_volume'] = data['volume'].rolling(10).apply(lambda x: x[:5].mean() if len(x) >= 5 else np.nan)
    data['pm_volume'] = data['volume'].rolling(10).apply(lambda x: x[5:].mean() if len(x) >= 10 else np.nan)
    data['volume_ratio'] = data['pm_volume'] / (data['am_volume'] + 1e-8)
    
    # Volume acceleration
    data['volume_acceleration'] = data['volume'].pct_change(periods=3)
    data['volume_spike'] = (data['volume'] > data['volume'].rolling(10).mean() * 1.5).astype(int)
    
    # Range Efficiency Metrics
    data['range_utilization'] = abs(data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    data['morning_range_ratio'] = (data['midday_price'] - data['low']) / (data['daily_range'] + 1e-8)
    data['afternoon_range_ratio'] = (data['high'] - data['midday_price']) / (data['daily_range'] + 1e-8)
    
    # Range-Volume Synchronization
    data['range_volume_corr'] = data['daily_range'].rolling(5).corr(data['volume'])
    data['range_efficiency'] = data['range_utilization'] * (1 + data['range_volume_corr'].fillna(0))
    
    # Momentum-Reversal Harmony Score
    data['momentum_reversal_alignment'] = np.where(
        (data['morning_momentum'] * data['afternoon_reversal']) > 0,
        data['morning_momentum_abs'] + abs(data['afternoon_reversal']),
        data['morning_momentum_abs'] - abs(data['afternoon_reversal'])
    )
    
    # Momentum Exhaustion Detection
    data['momentum_exhaustion'] = np.where(
        (data['morning_momentum_abs'] > data['morning_momentum_abs'].rolling(10).quantile(0.8)) &
        (data['volume_acceleration'] < 0) &
        (data['reversal_efficiency'] < data['reversal_efficiency'].rolling(10).median()),
        -1, 0
    )
    
    # Primary Signal Generation
    data['primary_signal'] = np.where(
        data['momentum_exhaustion'] == -1,
        -data['momentum_reversal_alignment'],  # Reversal signal
        np.where(
            (data['morning_momentum'] * data['afternoon_reversal'] > 0) & 
            (data['volume_ratio'] > 1) & 
            (data['range_efficiency'] > data['range_efficiency'].rolling(10).median()),
            data['momentum_reversal_alignment'],  # Momentum continuation
            0.5 * data['momentum_reversal_alignment']  # Range-bound
        )
    )
    
    # Confidence Adjustment
    data['efficiency_confidence'] = data['range_efficiency'] * (1 + data['range_volume_corr'].fillna(0))
    data['volume_confidence'] = np.where(
        data['volume_spike'] == 1,
        1.2,
        np.where(data['volume_acceleration'] > 0, 1.1, 0.9)
    )
    
    # Composite Alpha Factor
    data['composite_alpha'] = (
        data['primary_signal'] * 
        data['efficiency_confidence'] * 
        data['volume_confidence'] *
        (1 + 0.5 * data['reversal_efficiency'])
    )
    
    # Final factor with smoothing
    factor = data['composite_alpha'].rolling(3, min_periods=1).mean()
    
    return factor
