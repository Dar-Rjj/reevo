import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns for reference
    data['prev_close'] = data['close'].shift(1)
    data['ret'] = data['close'] / data['prev_close'] - 1
    
    # 1. Intraday Pressure Reversal
    data['pressure_index'] = ((data['close'] - data['low']) / 
                             (data['high'] - data['low'] + 1e-8) * data['volume'])
    
    # Calculate cross-sectional percentiles for pressure index
    pressure_percentiles = data.groupby(data.index)['pressure_index'].transform(
        lambda x: x.rank(pct=True)
    )
    data['extreme_pressure'] = (pressure_percentiles > 0.8).astype(int)
    
    # Reversal signal: next day (Open - Close) for extreme pressure stocks
    data['next_open'] = data['open'].shift(-1)
    data['reversal_signal'] = np.where(
        data['extreme_pressure'] == 1,
        data['next_open'] - data['close'],
        0
    )
    
    # 2. Amount-Momentum Persistence
    data['momentum'] = data['close'] / data['prev_close'] - 1
    data['amount_momentum'] = data['amount'] * data['momentum']
    
    # Calculate cross-sectional percentiles for amount-momentum
    am_percentiles = data.groupby(data.index)['amount_momentum'].transform(
        lambda x: x.rank(pct=True)
    )
    data['high_amount_momentum'] = (am_percentiles > 0.7).astype(int)
    
    # Persistence signal: next period momentum for high amount-momentum stocks
    data['next_ret'] = data['ret'].shift(-1)
    data['persistence_signal'] = np.where(
        data['high_amount_momentum'] == 1,
        data['next_ret'],
        0
    )
    
    # 3. Opening Shock Absorption
    data['shock_magnitude'] = data['open'] / data['prev_close'] - 1
    data['absorption_capacity'] = (
        (data['high'] - data['low']) / 
        (abs(data['open'] - data['prev_close']) + 1e-8)
    )
    data['absorption_signal'] = (
        (data['close'] - data['open']) / 
        (abs(data['shock_magnitude']) + 1e-8)
    )
    
    # 4. Volatility-Regime Switching
    # Calculate daily volatility
    data['daily_vol'] = (data['high'] - data['low']) / data['prev_close']
    
    # Rolling median volatility (5-day window)
    data['vol_median'] = data['daily_vol'].rolling(window=5, min_periods=3).median()
    data['vol_break'] = data['daily_vol'] > data['vol_median']
    
    # Rolling median volume (5-day window)
    data['vol_median_volume'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['volume_break'] = data['volume'] > data['vol_median_volume']
    
    # Regime signal: next day returns after volatility-volume co-break
    data['regime_signal'] = np.where(
        (data['vol_break'] == True) & (data['volume_break'] == True),
        data['next_ret'],
        0
    )
    
    # 5. Price-Range Efficiency
    data['range_utilization'] = (
        abs(data['close'] - data['open']) / 
        (data['high'] - data['low'] + 1e-8)
    )
    data['efficiency_score'] = (
        data['range_utilization'] * 
        (data['close'] - data['open']) / data['prev_close']
    )
    
    # Calculate cross-sectional percentiles for efficiency score
    eff_percentiles = data.groupby(data.index)['efficiency_score'].transform(
        lambda x: x.rank(pct=True)
    )
    data['high_efficiency'] = (eff_percentiles > 0.7).astype(int)
    
    # Efficiency signal: next day returns for high efficiency movers
    data['efficiency_signal'] = np.where(
        data['high_efficiency'] == 1,
        data['next_ret'],
        0
    )
    
    # Combine all signals with equal weights
    signals = [
        'reversal_signal', 'persistence_signal', 'absorption_signal', 
        'regime_signal', 'efficiency_signal'
    ]
    
    # Calculate final factor value (avoiding lookahead bias)
    for date in data.index:
        current_data = data.loc[date]
        factor_value = 0
        for signal in signals:
            if not pd.isna(current_data[signal]):
                factor_value += current_data[signal]
        factor.loc[date] = factor_value / len(signals)
    
    return factor
