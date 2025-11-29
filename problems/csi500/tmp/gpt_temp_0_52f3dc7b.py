import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate first 2 hours and last 2 hours data (assuming 6.5 hour trading day)
    # For simplicity, we'll approximate with first 30% and last 30% of the day
    data['High_First2H'] = data['high'].rolling(window=3, min_periods=1).apply(lambda x: x[:int(len(x)*0.3)].max() if len(x) >= 3 else np.nan)
    data['Low_First2H'] = data['low'].rolling(window=3, min_periods=1).apply(lambda x: x[:int(len(x)*0.3)].min() if len(x) >= 3 else np.nan)
    data['High_Last2H'] = data['high'].rolling(window=3, min_periods=1).apply(lambda x: x[-int(len(x)*0.3):].max() if len(x) >= 3 else np.nan)
    data['Low_Last2H'] = data['low'].rolling(window=3, min_periods=1).apply(lambda x: x[-int(len(x)*0.3):].min() if len(x) >= 3 else np.nan)
    data['Volume_First2H'] = data['volume'].rolling(window=3, min_periods=1).apply(lambda x: x[:int(len(x)*0.3)].sum() if len(x) >= 3 else np.nan)
    data['Volume_Last2H'] = data['volume'].rolling(window=3, min_periods=1).apply(lambda x: x[-int(len(x)*0.3):].sum() if len(x) >= 3 else np.nan)
    
    # Price Reversal Pattern Detection
    data['Morning_Return'] = (data['High_First2H'] - data['open']) / data['open']
    data['Afternoon_Return'] = (data['close'] - data['Low_Last2H']) / data['Low_Last2H']
    data['Extreme_Return_Difference'] = data['Morning_Return'] - data['Afternoon_Return']
    
    data['Morning_Strength'] = (data['High_First2H'] - data['open']) / (data['High_First2H'] - data['Low_First2H']).replace(0, np.nan)
    data['Afternoon_Strength'] = (data['close'] - data['Low_Last2H']) / (data['High_Last2H'] - data['Low_Last2H']).replace(0, np.nan)
    data['Reversal_Ratio'] = data['Morning_Strength'] / data['Afternoon_Strength'].replace(0, np.nan)
    
    data['Extreme_Reversal_Signal'] = data['Extreme_Return_Difference'] * data['Reversal_Ratio']
    data['Strength_Based_Reversal'] = data['Morning_Strength'] - data['Afternoon_Strength']
    data['Combined_Reversal'] = data['Extreme_Reversal_Signal'] * data['Strength_Based_Reversal']
    
    # Volume Confirmation Analysis
    data['Morning_Volume_Intensity'] = data['Volume_First2H'] / data['Volume_Last2H'].replace(0, np.nan)
    data['Afternoon_Volume_Momentum'] = data['Volume_Last2H'] / data['Volume_First2H'].replace(0, np.nan)
    data['Volume_Pattern_Ratio'] = data['Morning_Volume_Intensity'] / data['Afternoon_Volume_Momentum'].replace(0, np.nan)
    
    data['Morning_Alignment'] = data['Morning_Return'] * data['Morning_Volume_Intensity']
    data['Afternoon_Alignment'] = data['Afternoon_Return'] * data['Afternoon_Volume_Momentum']
    data['Volume_Price_Divergence'] = data['Morning_Alignment'] - data['Afternoon_Alignment']
    
    data['Volume_Confirmed_Reversal'] = data['Combined_Reversal'] * data['Volume_Price_Divergence']
    data['Pattern_Strength_Confirmation'] = data['Volume_Pattern_Ratio'] * data['Strength_Based_Reversal']
    data['Enhanced_Confirmation'] = data['Volume_Confirmed_Reversal'] * data['Pattern_Strength_Confirmation']
    
    # Price Level and Support Analysis
    data['Daily_Range_Position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['Morning_Range_Position'] = (data['High_First2H'] - data['Low_First2H']) / (data['high'] - data['low']).replace(0, np.nan)
    data['Position_Shift'] = data['Daily_Range_Position'] - data['Morning_Range_Position']
    
    data['Support_Level'] = data['low'] / data['close'].shift(1).replace(0, np.nan)
    data['Resistance_Level'] = data['high'] / data['close'].shift(1).replace(0, np.nan)
    data['Breakout_Potential'] = data['Resistance_Level'] - data['Support_Level']
    
    data['Position_Based_Reversal'] = data['Position_Shift'] * data['Combined_Reversal']
    data['Support_Enhanced_Signal'] = data['Breakout_Potential'] * data['Volume_Confirmed_Reversal']
    data['Level_Integrated_Factor'] = data['Position_Based_Reversal'] * data['Support_Enhanced_Signal']
    
    # Multi-Period Momentum Context
    data['Short_term_Momentum'] = data['close'] / data['close'].shift(2).replace(0, np.nan) - 1
    data['Medium_term_Momentum'] = data['close'] / data['close'].shift(5).replace(0, np.nan) - 1
    data['Momentum_Ratio'] = data['Short_term_Momentum'] / data['Medium_term_Momentum'].replace(0, np.nan)
    
    data['Momentum_Reversal'] = -1 * data['Short_term_Momentum'] * data['Medium_term_Momentum']
    data['Ratio_Based_Reversal'] = data['Momentum_Ratio'] * data['Combined_Reversal']
    data['Contextual_Reversal'] = data['Momentum_Reversal'] * data['Ratio_Based_Reversal']
    
    data['Momentum_Enhanced_Factor'] = data['Level_Integrated_Factor'] * data['Contextual_Reversal']
    data['Ratio_Weighted_Signal'] = data['Momentum_Enhanced_Factor'] * data['Momentum_Ratio']
    data['Final_Momentum_Integration'] = data['Ratio_Weighted_Signal'] * data['Momentum_Reversal']
    
    # Volatility and Range Dynamics
    data['Daily_Range_Volatility'] = (data['high'] - data['low']) / data['close'].replace(0, np.nan)
    data['Morning_Range_Volatility'] = (data['High_First2H'] - data['Low_First2H']) / data['open'].replace(0, np.nan)
    data['Volatility_Compression'] = data['Daily_Range_Volatility'] / data['Morning_Range_Volatility'].replace(0, np.nan)
    
    data['Range_Expansion'] = data['Daily_Range_Volatility'] - data['Daily_Range_Volatility'].shift(1)
    data['Compression_Signal'] = data['Volatility_Compression'] * data['Range_Expansion']
    data['Volatility_Context'] = data['Compression_Signal'] * data['Daily_Range_Volatility']
    
    data['Volatility_Adjusted_Factor'] = data['Final_Momentum_Integration'] * data['Volatility_Context']
    data['Range_Enhanced_Signal'] = data['Volatility_Adjusted_Factor'] * data['Daily_Range_Volatility']
    data['Compression_Weighted_Alpha'] = data['Range_Enhanced_Signal'] * data['Volatility_Compression']
    
    # Composite Alpha Construction
    data['Volume_Confirmed_Core'] = data['Enhanced_Confirmation'] * data['Level_Integrated_Factor']
    data['Momentum_Enhanced_Core'] = data['Volume_Confirmed_Core'] * data['Final_Momentum_Integration']
    
    data['Volatility_Weighted_Signal'] = data['Momentum_Enhanced_Core'] * data['Compression_Weighted_Alpha']
    data['Range_Adapted_Factor'] = data['Volatility_Weighted_Signal'] * data['Daily_Range_Volatility']
    
    data['Position_Enhanced_Alpha'] = data['Range_Adapted_Factor'] * data['Position_Shift']
    data['Support_Refined_Factor'] = data['Position_Enhanced_Alpha'] * data['Breakout_Potential']
    
    data['Volume_Final_Adjustment'] = data['Support_Refined_Factor'] * data['Volume_Pattern_Ratio']
    data['Intraday_Momentum_Reversal_Alpha'] = data['Volume_Final_Adjustment'] * data['Combined_Reversal']
    
    # Return the final alpha factor
    return data['Intraday_Momentum_Reversal_Alpha']
