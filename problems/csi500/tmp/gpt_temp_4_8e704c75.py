import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel intraday alpha factors using price, volume, and amount data.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Momentum Divergence
    data['price_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_momentum'] = data['volume_momentum'].fillna(0)
    factor1 = data['price_momentum'] * data['volume_momentum']
    
    # Factor 2: Range Breakout Efficiency
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['breakout_high'] = (data['close'] > data['prev_high']).astype(int)
    data['breakout_low'] = (data['close'] < data['prev_low']).astype(int)
    data['efficiency_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    factor2 = (data['breakout_high'] - data['breakout_low']) * data['efficiency_ratio']
    
    # Factor 3: Price Acceleration Divergence
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_acceleration'] = data['ret_3d'] - data['ret_10d']
    data['volume_trend'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean() - 1
    factor3 = data['price_acceleration'] * data['volume_trend']
    
    # Factor 4: Gap Mean Reversion
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['liquidity_proxy'] = data['amount'] / (data['high'] - data['low'] + 1e-8)
    factor4 = -data['opening_gap'] * data['liquidity_proxy']
    
    # Factor 5: Volatility-Regime Momentum
    data['intraday_range'] = data['high'] - data['low']
    data['volatility_ratio'] = data['intraday_range'] / data['intraday_range'].rolling(window=20, min_periods=1).mean()
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    factor5 = data['volatility_ratio'] * data['intraday_momentum']
    
    # Factor 6: Price-Volume Correlation Breakout
    data['price_ret'] = data['close'].pct_change()
    data['volume_ret'] = data['volume'].pct_change()
    
    # Calculate rolling correlation using only past data
    corr_values = []
    for i in range(len(data)):
        if i < 4:
            corr_values.append(0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            corr = window_data['price_ret'].corr(window_data['volume_ret'])
            corr_values.append(corr if not np.isnan(corr) else 0)
    
    data['price_volume_corr'] = corr_values
    
    data['range_spike'] = data['intraday_range'] / data['intraday_range'].rolling(window=5, min_periods=1).mean() - 1
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean() - 1
    factor6 = data['price_volume_corr'] * (data['range_spike'] + data['volume_spike'])
    
    # Combine factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5,
        'factor6': factor6
    })
    
    # Final factor is the average of all individual factors
    final_factor = factors.mean(axis=1, skipna=True)
    
    return final_factor
