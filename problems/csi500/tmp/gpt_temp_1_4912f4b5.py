import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Volatility-Weighted Gap Momentum Efficiency Factor
    """
    data = df.copy()
    
    # Multi-Timeframe Gap Momentum Structure
    # Intraday Gap Momentum Efficiency
    data['intraday_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_momentum_combined'] = data['intraday_momentum'] * np.abs(data['overnight_gap'])
    
    # Dual Gap Momentum Acceleration
    data['gap_5d_std'] = data['overnight_gap'].rolling(window=5).std()
    data['gap_3d_persistence'] = data['overnight_gap'].rolling(window=3).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 and not np.isnan(x).any() else np.nan
    )
    data['dual_gap_momentum'] = data['gap_momentum_combined'] * (1 + data['gap_5d_std']) * (1 + np.abs(data['gap_3d_persistence']))
    
    # Volatility-Adjusted Gap Range Analysis
    # Gap Volatility Persistence Integration
    data['volatility_ratio'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['gap_to_range_ratio'] = np.abs(data['overnight_gap']) / (data['high'] - data['low'])
    data['volatility_weighted_gap'] = data['dual_gap_momentum'] * data['volatility_ratio'] * data['gap_to_range_ratio']
    
    # Gap Range Compression Efficiency
    data['range_2d'] = data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    data['range_ratio_2d_1d'] = data['range_2d'] / (data['high'] - data['low'])
    data['price_gap_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Volume Context Integration
    data['volume_percentile_20d'] = data['volume'].rolling(window=20).apply(
        lambda x: (x[-1] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) > np.nanmin(x) else 0.5
    )
    data['gap_range_efficiency'] = data['price_gap_efficiency'] * data['volume_percentile_20d'] * data['range_ratio_2d_1d']
    
    # Amount-Confirmed Gap Dynamics
    # Opening Gap Analysis with Amount
    data['amount_trend'] = np.sign(data['amount'] - data['amount'].shift(1))
    data['gap_amount_combined'] = np.abs(data['overnight_gap']) * data['amount_trend']
    
    # Amount-Weighted Gap Absorption
    data['range_utilization'] = (data['high'] - data['low']) / (np.abs(data['overnight_gap']) + 1e-8)
    data['amount_acceleration'] = data['amount'] / data['amount'].shift(1) - 1
    data['amount_10d_avg'] = data['amount'].rolling(window=10).mean()
    data['amount_breakout'] = data['amount'] / data['amount_10d_avg'] - 1
    
    data['amount_confirmed_gap'] = data['gap_amount_combined'] * data['amount_acceleration'] * data['amount_breakout'] * data['range_utilization']
    
    # Volatility-Regime Gap Momentum
    # Identify Gap Volatility Context
    data['range_volatility_10d'] = (data['high'] - data['low']).rolling(window=10).std()
    data['volatility_regime'] = (data['range_volatility_10d'] > data['range_volatility_10d'].rolling(window=20).mean()).astype(int)
    
    # Evaluate Gap Reversal Efficiency
    data['gap_reversal'] = np.sign(data['overnight_gap'] * data['overnight_gap'].shift(1))
    data['gap_reversal_efficiency'] = data['price_gap_efficiency'] * (1 + data['gap_reversal']) * data['amount_breakout']
    
    data['volatility_regime_gap'] = data['volatility_weighted_gap'] * (1 + data['volatility_regime'] * 0.5) * data['gap_reversal_efficiency']
    
    # Composite Gap Factor Integration
    # Volatility-Persistent Gap Momentum Core
    data['core_gap_momentum'] = data['volatility_regime_gap'] * data['dual_gap_momentum'] * (1 + data['volatility_ratio'])
    
    # Gap Range Expansion Amplification
    data['gap_range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    data['range_expansion_amplified'] = data['core_gap_momentum'] * (1 + np.abs(data['gap_range_expansion']))
    
    # Amount Breakout Gap Confirmation
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['amount_breakout_strength'] = data['amount'] / data['amount_5d_avg'] - 1
    data['amount_price_trend'] = np.sign(data['overnight_gap'] * data['amount_breakout_strength'])
    
    data['breakout_confirmed_gap'] = data['range_expansion_amplified'] * (1 + data['amount_breakout_strength']) * (1 + data['amount_price_trend'])
    
    # Final Gap Factor Integration
    data['final_gap_factor'] = (
        data['breakout_confirmed_gap'] * 
        data['gap_range_efficiency'] * 
        data['amount_confirmed_gap'] * 
        (1 + data['gap_range_expansion']) * 
        (1 + data['volatility_regime'] * 0.3)
    )
    
    # Cross-sectional normalization
    factor = data.groupby(data.index)['final_gap_factor'].transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
