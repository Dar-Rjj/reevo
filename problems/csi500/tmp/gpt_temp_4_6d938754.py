import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Measurement
    data['Intraday_Efficiency_Ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Price_Slippage_Component'] = (data['high'] + data['low']) / 2 - (data['open'] + data['close']) / 2
    data['Efficiency_Adjusted_Momentum'] = data['Intraday_Efficiency_Ratio'] * data['Price_Slippage_Component']
    
    # Volume Microstructure Analysis - simplified proxies
    # Using daily volume patterns as proxy for intraday patterns
    data['Volume_Concentration_Pattern'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Price-Volume Divergence proxy using daily data
    data['daily_return'] = data['close'].pct_change()
    data['Price_Volume_Correlation'] = data['daily_return'].rolling(window=5, min_periods=1).corr(data['volume'].pct_change())
    data['Divergence_Signal'] = np.sign(data['close'] - data['open']) * data['Price_Volume_Correlation']
    data['Microstructure_Momentum'] = data['Volume_Concentration_Pattern'] * data['Divergence_Signal']
    
    # Order Imbalance and Flow Asymmetry
    data['Upward_Pressure_Signal'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['Downward_Pressure_Signal'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['Net_Pressure'] = data['Upward_Pressure_Signal'] - data['Downward_Pressure_Signal']
    
    # Buy-Sell Imbalance proxy using amount data
    data['amount_change'] = data['amount'].pct_change()
    data['Buy_Sell_Imbalance'] = data['amount'] / data['amount'].rolling(window=5, min_periods=1).mean()
    data['Asymmetry_Score'] = data['Net_Pressure'] * data['Buy_Sell_Imbalance']
    data['Imbalance_Momentum'] = data['Asymmetry_Score'] * data['Microstructure_Momentum']
    
    # Price Discovery and Information Flow
    # Midday price proxy using average of open and high/low midpoint
    data['midday_price'] = (data['open'] + (data['high'] + data['low']) / 2) / 2
    data['Early_vs_Late_Session_Performance'] = (data['close'] - data['midday_price']) / (data['midday_price'] - data['open'])
    data['Early_vs_Late_Session_Performance'] = data['Early_vs_Late_Session_Performance'].replace([np.inf, -np.inf], 0)
    data['Information_Efficiency'] = np.abs(data['Early_vs_Late_Session_Performance'])
    
    # Discovery Concentration proxy
    data['range'] = data['high'] - data['low']
    data['Discovery_Concentration'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    data['Timing_Efficiency'] = data['Discovery_Concentration'] * data['Information_Efficiency']
    data['Discovery_Momentum'] = data['Timing_Efficiency'] * data['Imbalance_Momentum']
    
    # Market Microstructure Regimes
    data['Volume_Stability'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).median()
    data['Price_Stability'] = data['range'] / data['range'].rolling(window=20, min_periods=1).mean()
    
    data['High_Liquidity_Multiplier'] = 1 + np.where(data['Volume_Stability'] > 1, data['Volume_Stability'], 0)
    data['Low_Volatility_Multiplier'] = 1 + np.where(data['Price_Stability'] < 0.5, data['Price_Stability'], 0)
    
    data['Regime_Enhanced_Signal'] = data['Discovery_Momentum'] * data['High_Liquidity_Multiplier'] * data['Low_Volatility_Multiplier']
    
    # Cross-Sectional Microstructure Patterns
    data['Efficiency_Rank'] = data['Intraday_Efficiency_Ratio'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    data['Volume_Rank'] = data['Volume_Concentration_Pattern'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    data['Combined_Rank'] = data['Efficiency_Rank'] * data['Volume_Rank']
    
    # Microstructure Momentum Persistence
    data['Microstructure_Momentum_3d_change'] = data['Regime_Enhanced_Signal'] - data['Regime_Enhanced_Signal'].shift(3)
    data['Persistence_Score'] = np.sign(data['Regime_Enhanced_Signal']) * np.sign(data['Microstructure_Momentum_3d_change'])
    
    data['Cross_Sectional_Factor'] = data['Combined_Rank'] * data['Persistence_Score'] * data['Regime_Enhanced_Signal']
    
    # Final Factor Construction
    data['Factor_Smoothing'] = data['Cross_Sectional_Factor'].rolling(window=2, min_periods=1).mean()
    
    # 5-day return for final adjustment
    data['5d_return'] = data['close'].pct_change(5)
    data['Final_Factor'] = data['Factor_Smoothing'] * np.sign(data['5d_return'])
    
    # Handle NaN values
    data['Final_Factor'] = data['Final_Factor'].fillna(0)
    
    return data['Final_Factor']
