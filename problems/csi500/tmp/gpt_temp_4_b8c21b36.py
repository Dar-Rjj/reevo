import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 6:
        return factor
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['range'].shift(1)
    
    # Rolling windows
    data['vol_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['range_5d_avg'] = data['range'].rolling(window=5, min_periods=3).mean()
    
    # Gap-Range Efficiency Divergence components
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['range_efficiency'] = (data['close'] - data['open']) / (data['range'] + 1e-8)
    data['range_pressure'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['range'] + 1e-8)
    data['volume_confirmation'] = data['volume'] / (data['vol_5d_avg'] + 1e-8)
    
    # Volatility-Adaptive Compression Breakout components
    data['compression_ratio'] = data['range'] / (data['range_5d_avg'] + 1e-8)
    data['price_acceleration'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    data['volume_acceleration'] = data['volume'] / (data['vol_5d_avg'] + 1e-8)
    data['compression_regime'] = (data['range'] < data['range_5d_avg']).astype(int)
    
    # Efficiency-Weighted Reversal components
    data['local_min'] = data['close'].rolling(window=3, min_periods=3).min()
    data['local_max'] = data['close'].rolling(window=3, min_periods=3).max()
    data['gap_fade'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    data['flow_efficiency'] = data['amount'] / (data['close'] * data['volume'] + 1e-8)
    
    # Intraday Session Divergence components
    data['morning_momentum'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['afternoon_momentum'] = (data['close'] - data['low']) / (data['low'] + 1e-8)
    data['net_pressure'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['range'] + 1e-8)
    
    # Estimate volume skew (assuming morning volume = first half of day, afternoon = second half)
    # Since we only have daily aggregates, we'll use price-based proxy
    data['volume_skew'] = (data['open'] - data['low']) / (data['high'] - data['close'] + 1e-8)
    
    # Calculate composite factors
    for i in range(5, len(data)):
        if not (pd.isna(data.iloc[i]['vol_5d_avg']) or pd.isna(data.iloc[i]['range_5d_avg'])):
            
            # Gap-Range Efficiency Divergence Factor
            gap_range_divergence = (
                data.iloc[i]['gap_momentum'] * 
                data.iloc[i]['range_efficiency'] * 
                data.iloc[i]['range_pressure'] * 
                data.iloc[i]['volume_confirmation']
            )
            
            # Volatility-Adaptive Compression Breakout Factor
            compression_breakout = (
                (1 - data.iloc[i]['compression_ratio']) * 
                data.iloc[i]['price_acceleration'] * 
                data.iloc[i]['volume_acceleration'] * 
                (1 if data.iloc[i]['compression_regime'] else -1)
            )
            
            # Efficiency-Weighted Reversal Factor
            reversal_efficiency = (
                (data.iloc[i]['close'] - data.iloc[i]['local_min']) / (data.iloc[i]['local_max'] - data.iloc[i]['local_min'] + 1e-8) * 
                data.iloc[i]['gap_fade'] * 
                data.iloc[i]['flow_efficiency']
            )
            
            # Intraday Session Divergence Factor
            session_divergence = (
                data.iloc[i]['morning_momentum'] * 
                data.iloc[i]['afternoon_momentum'] * 
                data.iloc[i]['net_pressure'] * 
                data.iloc[i]['volume_skew']
            )
            
            # Final composite factor (equal weighted combination)
            factor.iloc[i] = (
                gap_range_divergence + 
                compression_breakout + 
                reversal_efficiency + 
                session_divergence
            ) / 4
    
    # Handle any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
