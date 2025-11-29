import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Regime Adjusted Intraday Pressure
    # Calculate Intraday Pressure Components
    df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate Net Pressure Accumulation
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    df['accumulated_pressure'] = df['net_pressure'].rolling(window=3, min_periods=1).sum()
    
    # Determine Volatility Regime
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(window=20, min_periods=1).std()
    df['volatility_50d_avg'] = df['volatility_20d'].rolling(window=50, min_periods=1).mean()
    df['volatility_regime'] = np.where(df['volatility_20d'] > df['volatility_50d_avg'], 1, -1)
    
    # Adjust Pressure by Regime
    df['trading_intensity'] = df['volume'] * df['amount']
    factor1 = df['accumulated_pressure'] * df['volatility_regime'] * df['trading_intensity']
    
    # Amount-Weighted Gap Reversal Efficiency
    # Calculate Gap Components
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Efficiency Metrics
    df['true_range'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                           abs(df['low'] - df['close'].shift(1))))
    df['price_range_efficiency'] = (df['high'] - df['low']) / df['true_range'].replace(0, np.nan)
    
    # Apply Amount-Based Weighting
    df['amount_rank'] = df['amount'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['amount_weight'] = 1 - df['amount_rank']
    
    # Combine Efficiency and Reversal
    gap_fill_probability = -df['overnight_gap']  # Inverse for mean reversion
    factor2 = gap_fill_probability * df['price_range_efficiency'] * df['amount_weight'] * (df['high'] - df['low'])
    
    # Momentum Acceleration with Volume Divergence
    # Calculate Momentum Components
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    
    # Calculate Acceleration Signal
    df['acceleration'] = df['momentum_5d'] - df['momentum_10d']
    
    # Detect Volume Confirmation
    df['volume_5d_change'] = df['volume'].pct_change(5)
    df['volume_divergence'] = df['volume_5d_change'] - df['volume_5d_change'].rolling(window=20, min_periods=1).mean()
    
    # Combine Acceleration and Volume
    df['intraday_amplitude'] = (df['high'] - df['low']) / df['open'].replace(0, np.nan)
    factor3 = df['acceleration'] * df['volume_divergence'] * df['intraday_amplitude']
    
    # Range-Adjusted Opening Momentum Persistence
    # Calculate Opening Momentum
    df['high_open_momentum'] = df['high'] - df['open']
    df['close_low_momentum'] = df['close'] - df['low']
    df['opening_signal'] = (df['high_open_momentum'] + df['close_low_momentum']) / 2
    
    # Assess Persistence Strength
    df['price_change'] = df['close'].pct_change()
    df['persistence_count'] = 0
    for i in range(1, 4):
        df['persistence_count'] += ((df['price_change'].shift(i) > 0) & (df['price_change'] > 0)) | \
                                  ((df['price_change'].shift(i) < 0) & (df['price_change'] < 0))
    
    # Compute Range-Based Adjustment
    df['range_efficiency'] = (df['high'] - df['low']) / df['true_range'].replace(0, np.nan)
    df['adjusted_momentum'] = df['opening_signal'] * df['range_efficiency'] * df['persistence_count']
    
    # Generate Persistence Prediction
    df['amplitude_weight'] = (df['high'] - df['low']) / df['open'].replace(0, np.nan)
    factor4 = df['adjusted_momentum'] * df['amplitude_weight'] * df['trading_intensity']
    
    # Combine all factors with equal weighting
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + factor4.fillna(0)) / 4
    
    return combined_factor
