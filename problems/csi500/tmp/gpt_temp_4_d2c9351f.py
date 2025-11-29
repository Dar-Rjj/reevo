import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Calculate intraday return extremes
    data['intraday_gain'] = (data['high'] - data['open']) / data['open']
    data['intraday_loss'] = (data['low'] - data['open']) / data['open']
    
    # Calculate maximum intraday moves
    data['max_gain'] = data['intraday_gain']
    data['max_loss'] = data['intraday_loss']
    
    # Calculate close position relative to daily range
    data['daily_range'] = data['high'] - data['low']
    data['close_to_low_ratio'] = (data['close'] - data['low']) / data['daily_range']
    data['high_to_close_ratio'] = (data['high'] - data['close']) / data['daily_range']
    
    # Calculate volume moving average
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5']
    
    # Calculate gain/loss percentiles for threshold determination
    data['gain_percentile'] = data['intraday_gain'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['loss_percentile'] = data['intraday_loss'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Identify reversal signals
    # Bearish reversal: large gain followed by close near low
    bearish_signal = ((data['gain_percentile'] >= 0.8) &  # top 20% gains
                      (data['close_to_low_ratio'] < 0.3))  # close near low
    
    # Bullish reversal: large loss followed by close near high  
    bullish_signal = ((data['loss_percentile'] <= 0.2) &  # bottom 20% losses
                      (data['high_to_close_ratio'] < 0.3))  # close near high
    
    # Create signal values (negative for bearish, positive for bullish)
    signal_strength = np.zeros(len(data))
    signal_strength[bearish_signal] = -1
    signal_strength[bullish_signal] = 1
    
    # Apply volume confirmation
    factor_values = signal_strength * data['volume_ratio']
    
    return pd.Series(factor_values, index=data.index)
