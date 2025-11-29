import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Alpha Factor combining multiple gap-based reversal strategies with volume efficiency
    """
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate flow efficiency
    data['flow_efficiency'] = data['amount'] / (data['close'] * data['volume']).replace(0, np.nan)
    
    # Gap-Momentum Divergence Component
    data['gap_sign'] = np.sign(data['gap'])
    data['momentum_sign'] = np.sign(data['intraday_momentum'])
    data['divergence'] = (data['gap_sign'] != data['momentum_sign']).astype(int)
    
    # Gap persistence tracking
    for window in [3, 5]:
        data[f'gap_persistence_{window}'] = data['gap_sign'].rolling(window=window).apply(
            lambda x: len(set(x)) == 1 if not x.isnull().any() else np.nan
        ).astype(float)
    
    # Volume distribution (simplified morning/afternoon ratio)
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Efficiency momentum
    data['efficiency_momentum'] = data['flow_efficiency'].pct_change(periods=5)
    
    # Compression detection
    data['price_range'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    data['range_median_10'] = data['price_range'].rolling(window=10).median()
    data['compression'] = (data['price_range'] < data['range_median_10']).astype(int)
    
    # Large gap detection
    gap_threshold = data['gap'].abs().rolling(window=20).quantile(0.7)
    data['large_gap'] = (data['gap'].abs() > gap_threshold).astype(int)
    
    # Reversal detection
    data['next_day_return'] = data['close'].pct_change().shift(-1)
    data['gap_reversal'] = (np.sign(data['gap']) != np.sign(data['next_day_return'])).astype(int)
    
    # Range pressure components
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    
    # Local extrema detection
    data['local_min'] = (data['close'] == data['close'].rolling(window=3, center=True).min()).astype(int)
    data['local_max'] = (data['close'] == data['close'].rolling(window=3, center=True).max()).astype(int)
    data['reversal_point'] = data['local_min'] | data['local_max']
    
    # Combined factor calculation
    # Component 1: Gap-Momentum Divergence
    comp1 = (data['divergence'] * 
             data['gap_persistence_3'].fillna(0) * 
             data['efficiency_momentum'].fillna(0) * 
             data['gap'].abs())
    
    # Component 2: Compression Breakout
    comp2 = (data['compression'] * 
             data['large_gap'] * 
             data['gap_reversal'].fillna(0) * 
             data['volume_ratio'].fillna(1) * 
             data['net_pressure'].abs())
    
    # Component 3: Range Reversal
    comp3 = (data['reversal_point'] * 
             data['true_range'].pct_change().fillna(0) * 
             data['efficiency_momentum'].fillna(0) * 
             data['divergence'])
    
    # Component 4: Intraday Pressure
    comp4 = (data['net_pressure'] * 
             data['gap_persistence_5'].fillna(0) * 
             data['volume_ratio'].fillna(1) * 
             data['intraday_momentum'].abs())
    
    # Adaptive oscillator component
    gap_ma_short = data['gap'].rolling(window=3).mean()
    gap_ma_long = data['gap'].rolling(window=8).mean()
    data['gap_oscillator'] = gap_ma_short - gap_ma_long
    
    # Final combined factor
    factor = (comp1.fillna(0) * 0.25 + 
              comp2.fillna(0) * 0.25 + 
              comp3.fillna(0) * 0.25 + 
              comp4.fillna(0) * 0.25 + 
              data['gap_oscillator'].fillna(0) * 0.1)
    
    # Volatility adjustment
    vol_adj = data['true_range'].rolling(window=10).std().fillna(1)
    factor = factor / vol_adj.replace(0, 1)
    
    # Remove any forward-looking data contamination
    factor = factor.shift(1)  # Ensure no lookahead
    
    return factor
