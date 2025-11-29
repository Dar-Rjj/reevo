import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Intraday Volatility Components
    data['Close_prev'] = data['close'].shift(1)
    data['True_Range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['Close_prev']),
            np.abs(data['low'] - data['Close_prev'])
        )
    )
    data['Normalized_True_Range'] = data['True_Range'] / data['open']
    data['Volatility_Clustering'] = data['Normalized_True_Range'].rolling(window=5, min_periods=3).std()
    
    # Calculate Liquidity Proxy Measures
    data['Amihud_Illiquidity'] = np.abs(data['close'] - data['Close_prev']) / data['amount']
    
    # Roll Spread Estimate using 5-day window
    def roll_spread(series):
        if len(series) < 2:
            return np.nan
        price_changes = series.diff().dropna()
        if len(price_changes) < 2:
            return np.nan
        cov = price_changes.cov(price_changes.shift(1))
        return 2 * np.sqrt(max(0, -cov)) if cov < 0 else 0
    
    data['Roll_Spread_Estimate'] = data['close'].rolling(window=5, min_periods=3).apply(roll_spread, raw=False)
    data['Volume_Weighted_Spread'] = (data['high'] - data['low']) / (data['volume'] * data['close'])
    
    # Construct Volatility-Liquidity Interactions
    data['Volatility_per_Unit_Spread'] = data['Normalized_True_Range'] / data['Roll_Spread_Estimate'].replace(0, np.nan)
    data['Liquidity_Absorption'] = data['volume'] / (data['Amihud_Illiquidity'] * data['close']).replace(0, np.nan)
    data['Efficiency_Adjusted_Volatility'] = data['Volatility_per_Unit_Spread'] * data['Liquidity_Absorption']
    
    # Generate Microstructure Noise Signals
    data['Price_Reversal_Component'] = (data['close'] - data['Close_prev']) / data['True_Range'].replace(0, np.nan)
    data['Bid_Ask_Bounce_Proxy'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']).replace(0, np.nan)
    data['Noise_Ratio'] = data['Price_Reversal_Component'] / data['Bid_Ask_Bounce_Proxy'].replace(0, np.nan)
    
    # Calculate Order Flow Imbalance Proxies
    data['Volume_Close_Divergence'] = data['volume'] * (data['close'] - (data['high'] + data['low'])/2)
    data['Amount_5day_mean'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['Amount_Based_Pressure'] = data['amount'] / data['Amount_5day_mean'].replace(0, np.nan)
    data['Flow_Imbalance'] = data['Volume_Close_Divergence'] * data['Amount_Based_Pressure']
    
    # Construct Regime-Dependent Signals
    data['Norm_TR_5day_median'] = data['Normalized_True_Range'].rolling(window=5, min_periods=3).median()
    data['Amihud_5day_median'] = data['Amihud_Illiquidity'].rolling(window=5, min_periods=3).median()
    data['High_Volatility_Regime'] = (data['Normalized_True_Range'] > data['Norm_TR_5day_median']).astype(float)
    data['Low_Liquidity_Regime'] = (data['Amihud_Illiquidity'] > data['Amihud_5day_median']).astype(float)
    data['Regime_Interaction'] = data['High_Volatility_Regime'] * data['Low_Liquidity_Regime']
    
    # Final Alpha Construction
    data['Volatility_Liquidity_Anomaly'] = data['Efficiency_Adjusted_Volatility'] * data['Flow_Imbalance']
    data['Regime_Weighted_Anomaly'] = data['Volatility_Liquidity_Anomaly'] * data['Regime_Interaction']
    data['Alpha_Factor'] = data['Regime_Weighted_Anomaly'].rolling(window=3, min_periods=2).mean()
    
    # Return the alpha factor series
    return data['Alpha_Factor']
