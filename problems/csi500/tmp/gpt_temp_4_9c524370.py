import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['intraday_range'] = data['high'] - data['low']
    data['abs_price_change'] = abs(data['close'] - data['prev_close'])
    
    # Intraday Pressure Components
    # Opening Pressure Signal
    data['opening_pressure'] = ((data['open'] - data['prev_close']) / 
                               (data['intraday_range'] + 0.001)) * data['abs_price_change']
    
    # Closing Pressure Signal
    data['closing_pressure'] = ((data['close'] - data['open']) / 
                               (data['intraday_range'] + 0.001)) * (data['intraday_range'] / (data['abs_price_change'] + 0.001))
    
    # Volume-Pressure Divergence
    data['log_volume'] = np.log(data['volume'] + 1)
    data['buying_pressure'] = ((data['close'] - data['low']) / (data['intraday_range'] + 0.001)) * data['log_volume']
    data['selling_pressure'] = ((data['high'] - data['close']) / (data['intraday_range'] + 0.001)) * data['log_volume']
    data['pressure_divergence'] = (data['buying_pressure'] - data['selling_pressure']) / (data['buying_pressure'] + data['selling_pressure'] + 0.001)
    
    # Volatility-Entropy Integration
    data['volatility_entropy'] = data['intraday_range'] / (abs(data['close'] - data['open']) + 0.001)
    
    # Calculate pressure-volatility correlation over 5-day window
    data['pressure_vol_corr'] = data['opening_pressure'].rolling(window=5).corr(data['volatility_entropy'])
    
    # Measure entropy persistence
    data['entropy_persistence'] = data['volatility_entropy'].rolling(window=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 else np.nan)
    
    # Range-Pressure Filtering
    data['range_median_10d'] = data['intraday_range'].rolling(window=10).median()
    data['abnormal_range'] = (data['intraday_range'] > 2 * data['range_median_10d']).astype(int)
    data['pressure_concentration'] = abs(data['close'] - data['open']) / (data['intraday_range'] + 0.001)
    
    # Multi-timeframe Pressure Alignment
    # 3-day rolling pressure direction consistency
    data['pressure_direction'] = np.sign(data['opening_pressure'])
    data['direction_consistency_3d'] = data['pressure_direction'].rolling(window=3).apply(lambda x: len(set(x)) == 1 if len(x) == 3 else np.nan)
    
    # 5-day pressure magnitude persistence
    data['pressure_magnitude_persistence'] = data['opening_pressure'].rolling(window=5).apply(lambda x: np.std(x) / (np.mean(abs(x)) + 0.001))
    
    # Inter-day pressure transmission efficiency
    data['pressure_transmission'] = data['opening_pressure'].rolling(window=3).apply(
        lambda x: abs(x.iloc[-1]) / (np.mean(abs(x.iloc[:-1])) + 0.001) if len(x) == 3 else np.nan
    )
    
    # Factor Synthesis
    # Combine pressure components
    pressure_combined = (data['opening_pressure'] + data['closing_pressure']) / 2
    
    # Apply volatility-entropy normalization
    volatility_normalized = pressure_combined / (data['volatility_entropy'] + 0.001)
    
    # Combine with volume divergence
    volume_enhanced = volatility_normalized * data['pressure_divergence']
    
    # Filter through range-pressure conditions
    range_filtered = volume_enhanced * (1 - data['abnormal_range']) * data['pressure_concentration']
    
    # Enhance with multi-timeframe alignment
    timeframe_enhanced = (range_filtered * 
                         data['direction_consistency_3d'].fillna(0) * 
                         (1 - data['pressure_magnitude_persistence'].fillna(0)) * 
                         data['pressure_transmission'].fillna(0))
    
    # Final factor as reversal probability
    factor = -timeframe_enhanced  # Negative for reversal signal
    
    return factor
