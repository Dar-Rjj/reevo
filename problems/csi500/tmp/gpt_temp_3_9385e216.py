import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Elasticity Components
    data['Opening_Elasticity'] = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    data['Closing_Elasticity'] = (data['close'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    data['Elasticity_Divergence'] = data['Opening_Elasticity'] - data['Closing_Elasticity']
    
    # Volatility Asymmetry and Exhaustion Signals
    data['Volatility_Asymmetry'] = ((data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) - \
                                  ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan))
    data['Exhaustion_Score'] = abs((data['high']/data['open'] - 1) - (data['low']/data['open'] - 1))
    data['Volatility_Breakout_Ratio'] = (data['high'] - data['low']) / data['high'].rolling(15).apply(lambda x: np.nanmedian(x.diff().abs()))
    
    # Volume Acceleration and Distribution Analysis
    data['Volume_Acceleration'] = data['volume'] / data['volume'].rolling(5).median()
    # Using first half volume approximation (assuming symmetric intraday distribution)
    data['Volume_Distribution_Asymmetry'] = (data['volume'] * 0.4 / data['volume']) - 0.5
    data['Volume_Momentum_Alignment'] = np.sign(data['close'] - data['open']) * data['Volume_Acceleration'] * \
                                       ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan))
    
    # Elasticity-Volatility Interaction
    data['Elasticity_Asymmetry_Interaction'] = data['Elasticity_Divergence'] * data['Volatility_Asymmetry']
    data['Exhaustion_Enhanced_Elasticity'] = data['Elasticity_Asymmetry_Interaction'] * data['Exhaustion_Score']
    data['Breakout_Weighted_Elasticity'] = data['Exhaustion_Enhanced_Elasticity'] * data['Volatility_Breakout_Ratio']
    
    # Volume Acceleration Confirmation
    data['Distribution_Weighted_Acceleration'] = data['Volume_Acceleration'] * data['Volume_Distribution_Asymmetry']
    data['Acceleration_Momentum_Alignment'] = data['Distribution_Weighted_Acceleration'] * data['Volume_Momentum_Alignment']
    
    # Final Composite Signal
    data['Raw_Alpha'] = data['Breakout_Weighted_Elasticity'] * data['Acceleration_Momentum_Alignment']
    
    # Dynamic Volatility Adjustment
    data['Volatility_Momentum'] = (data['high'] - data['low']) / data['high'].rolling(5).apply(lambda x: np.nanmedian(x.diff().abs())) - 1
    data['Regime_Multiplier'] = 1 + abs(data['Volatility_Momentum'])
    data['Final_Signal'] = data['Raw_Alpha'] * data['Regime_Multiplier'] - \
                          (data['Raw_Alpha'] * data['Regime_Multiplier']).rolling(5).median()
    
    return data['Final_Signal']
