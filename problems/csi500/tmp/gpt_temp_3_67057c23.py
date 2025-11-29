import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['returns'] = data['close'].pct_change()
    
    # Calculate True Range for ATR
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    
    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=1).sum() / data['volume'].rolling(window=1).sum()
    
    # Factor 1: Intraday Gap Absorption Factor
    data['gap_size'] = data['open'] / data['prev_close'] - 1
    data['gap_direction'] = np.sign(data['gap_size'])
    
    # Avoid division by zero
    gap_abs = abs(data['gap_size'])
    gap_abs = gap_abs.replace(0, np.nan)
    
    data['high_absorption'] = (data['high'] - data['open']) / gap_abs
    data['low_absorption'] = (data['open'] - data['low']) / gap_abs
    
    # Volume intensity
    data['volume_median_10'] = data['volume'].rolling(window=10).median()
    data['volume_intensity'] = data['volume'] / data['volume_median_10']
    
    # Combine gap absorption signals
    data['absorption_signal'] = np.where(data['gap_direction'] > 0, 
                                        data['high_absorption'], 
                                        data['low_absorption'])
    
    factor1 = data['absorption_signal'] * data['volume_intensity'] * data['gap_direction']
    
    # Factor 2: Volatility Regime Transition Factor
    data['vol_short'] = data['returns'].rolling(window=5).std()
    data['vol_long'] = data['returns'].rolling(window=20).std()
    data['vol_ratio'] = data['vol_short'] / data['vol_long']
    
    # Price efficiency
    data['daily_range'] = data['high'] - data['low']
    data['price_efficiency'] = data['daily_range'] / data['atr_5']
    
    # Volume change
    data['volume_change'] = data['volume'].pct_change()
    
    factor2 = data['vol_ratio'] * data['price_efficiency'] * data['volume_change']
    
    # Factor 3: Momentum Acceleration Factor
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    
    # Price acceleration
    numerator = data['return_3d'] - data['return_1d']
    denominator = data['return_5d'] - data['return_3d']
    data['price_acceleration'] = numerator / denominator.replace(0, np.nan)
    
    # Volume acceleration
    data['volume_slope_5'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['volume_slope_10'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    data['volume_acceleration'] = data['volume_slope_5'] / data['volume_slope_10'].replace(0, np.nan)
    
    factor3 = data['price_acceleration'] * data['volume_acceleration']
    
    # Factor 4: Intraday Pressure Build-up Factor
    data['pressure_index'] = (data['close'] - data['vwap']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Calculate pressure duration (consecutive periods with same pressure sign)
    data['pressure_sign'] = np.sign(data['pressure_index'])
    data['pressure_duration'] = data.groupby((data['pressure_sign'] != data['pressure_sign'].shift(1)).cumsum())['pressure_sign'].cumcount() + 1
    
    # Volume density
    data['volume_density'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    factor4 = data['pressure_index'] * data['pressure_duration'] * data['volume_density']
    
    # Factor 5: Range Expansion Quality Factor
    data['daily_range_avg_5'] = data['daily_range'].rolling(window=5).mean()
    data['range_expansion'] = data['daily_range'] / data['daily_range_avg_5']
    
    data['breakout_purity'] = abs(data['close'] - data['open']) / data['daily_range'].replace(0, np.nan)
    
    # Opening hour volume (approximated as first hour volume)
    data['opening_hour_volume'] = data['volume'].rolling(window=1).mean()  # Simplified approximation
    
    data['volume_profile'] = data['volume'] / data['opening_hour_volume'].replace(0, np.nan)
    
    factor5 = data['range_expansion'] * data['breakout_purity'] * data['volume_profile']
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
