import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Intraday Efficiency Component
    df['Day_Range_Efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['Volume_Weighted_Efficiency'] = df['Day_Range_Efficiency'] * df['volume']
    df['Efficiency_Momentum'] = df['Volume_Weighted_Efficiency'].rolling(window=3, min_periods=1).sum()
    
    # Volatility Structure Analysis
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['close']
    df['Volatility_Regime'] = df['Intraday_Volatility'] / df['Intraday_Volatility'].rolling(window=10, min_periods=1).mean()
    df['Gap_Absorption'] = abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Volume Dynamics Component
    df['Volume_Concentration'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).max()
    df['Volume_Acceleration'] = df['volume'] / df['volume'].shift(3) - 1
    df['Volume_Volatility_Ratio'] = df['volume'] / (df['high'] - df['low'])
    
    # Momentum-Reversal Analysis
    df['AM_Momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['PM_Reversal'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    
    def rolling_corr_3d(x, y):
        return pd.Series([x.iloc[max(0, i-2):i+1].corr(y.iloc[max(0, i-2):i+1]) 
                         for i in range(len(x))], index=x.index)
    
    df['Momentum_Reversal_Divergence'] = (abs(df['AM_Momentum'] - df['PM_Reversal']) * 
                                         rolling_corr_3d(df['AM_Momentum'], df['PM_Reversal']))
    
    # Support-Resistance Dynamics
    df['High_Break_Potential'] = (df['high'] - df['high'].rolling(window=5, min_periods=1).max()) / df['high'].rolling(window=5, min_periods=1).std()
    df['Low_Break_Potential'] = (df['low'].rolling(window=5, min_periods=1).min() - df['low']) / df['low'].rolling(window=5, min_periods=1).std()
    df['Breakout_Imbalance'] = df['High_Break_Potential'] - df['Low_Break_Potential']
    
    # Range Compression Assessment
    df['Range_Compression_Ratio'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    df['Position_Compression'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['Compression_Intensity'] = df['Range_Compression_Ratio'] * df['Position_Compression']
    
    # Price-Volume Integration
    df['Volume_Confirmed_Efficiency'] = df['Efficiency_Momentum'] * df['Volume_Concentration']
    df['Volatility_Adjusted_Volume'] = df['volume'] / df['Intraday_Volatility']
    df['Volume_Flow_Direction'] = np.sign((df['close'] - df['open']) * df['volume'])
    
    # Composite Signal Generation
    df['Core_Efficiency_Signal'] = df['Volume_Confirmed_Efficiency'] * df['Volatility_Regime']
    df['Momentum_Volatility_Composite'] = df['Momentum_Reversal_Divergence'] * df['Gap_Absorption']
    df['Breakout_Volume_Composite'] = df['Breakout_Imbalance'] * df['Volume_Acceleration']
    df['Compression_Momentum_Signal'] = df['AM_Momentum'] * df['Compression_Intensity']
    
    # Multi-Dimensional Confirmation
    df['Efficiency_Volatility_Core'] = df['Core_Efficiency_Signal'] * df['Momentum_Volatility_Composite']
    df['Volume_Breakout_Enhancement'] = df['Breakout_Volume_Composite'] * df['Volume_Flow_Direction']
    df['Compression_Momentum_Overlay'] = df['Compression_Momentum_Signal'] * df['Volatility_Adjusted_Volume']
    
    # Final Factor Integration
    df['Primary_Interaction'] = df['Efficiency_Volatility_Core'] * df['Volume_Breakout_Enhancement']
    df['Secondary_Confirmation'] = df['Primary_Interaction'] * df['Compression_Momentum_Overlay']
    df['Final_Factor'] = df['Secondary_Confirmation'] * df['PM_Reversal']
    
    return df['Final_Factor']
