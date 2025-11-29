import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Compute Intraday Return Extremes
    data['intraday_gain'] = data['high'] / data['open'] - 1
    data['intraday_loss'] = data['low'] / data['open'] - 1
    
    # Calculate Average True Range (ATR) for normalization
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    
    # Identify Momentum Exhaustion Signals
    data['gain_loss_diff'] = abs(data['intraday_gain']) - abs(data['intraday_loss'])
    data['normalized_exhaustion'] = data['gain_loss_diff'] / data['atr_5']
    
    # Create reversal signals based on extreme moves
    data['bullish_reversal_signal'] = np.where(data['intraday_loss'] < -0.02, 1, 0)
    data['bearish_reversal_signal'] = np.where(data['intraday_gain'] > 0.02, -1, 0)
    data['exhaustion_signal'] = data['bullish_reversal_signal'] + data['bearish_reversal_signal']
    
    # Volume-Based Confirmation
    data['volume_ma_3'] = data['volume'].rolling(window=3, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / data['volume_ma_3'] - 1
    
    # Calculate volume percentile rank over 20-day window
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Combine Price and Volume Signals
    data['volume_adjusted_signal'] = data['exhaustion_signal'] * data['volume_acceleration']
    data['composite_reversal_score'] = data['volume_adjusted_signal'] * data['volume_percentile']
    
    # Final factor value
    factor = data['composite_reversal_score']
    
    return factor
