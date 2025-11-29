import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['Close_prev'] = data['close'].shift(1)
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['Close_prev'])
    data['TR3'] = abs(data['low'] - data['Close_prev'])
    data['True_Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Volatility Breakout Detection
    data['Volatility_Breakout'] = (data['close'] - data['open']) / data['True_Range']
    data['Volatility_Expansion'] = data['True_Range'] / data['True_Range'].shift(1)
    
    # Price Path Efficiency
    data['Open_to_Close_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['High_Low_Capture'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['Path_Deviation'] = abs(data['Open_to_Close_Efficiency'] - data['High_Low_Capture'])
    
    # Volume Confirmation
    data['Volume_Intensity'] = data['volume'] / data['volume'].shift(1)
    data['Volume_Price_Alignment'] = np.sign(data['close'] - data['open']) * data['Volume_Intensity']
    
    # Multi-Timeframe Integration
    data['Short_term_Momentum'] = data['Volatility_Breakout'] - data['Volatility_Breakout'].shift(1)
    data['Medium_term_Consistency'] = data['Volatility_Breakout'].rolling(window=5, min_periods=3).mean()
    
    # Calculate Efficiency Momentum
    data['Efficiency_Momentum'] = data['Path_Deviation'] - data['Path_Deviation'].shift(1)
    
    # Combine components
    data['Volatility_Component'] = data['Volatility_Expansion'] * data['Volatility_Breakout']
    data['Raw_Factor'] = data['Volatility_Component'] * data['Efficiency_Momentum'] * data['Volume_Price_Alignment']
    
    # Cross-sectional ranking
    def cross_sectional_rank(group):
        return group.rank(pct=True)
    
    # Calculate cross-sectional ranks
    data['Factor_Rank'] = data.groupby(data.index)['Raw_Factor'].transform(cross_sectional_rank)
    
    # Final factor (normalized cross-sectional rank)
    factor = data['Factor_Rank'] - 0.5
    
    # Clean up intermediate columns
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    return factor
