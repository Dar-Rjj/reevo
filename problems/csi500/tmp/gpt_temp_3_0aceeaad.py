import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Intraday Volatility-Momentum Acceleration Factor
    Combines volatility expansion, momentum acceleration, and volume flow confirmation
    """
    df = data.copy()
    
    # Calculate Intraday Volatility Components
    df['intraday_range'] = df['high'] - df['low']
    df['open_to_low'] = df['open'] - df['low']
    df['high_to_open'] = df['high'] - df['open']
    
    # Compute Volatility Expansion Signal
    # Assuming first hour = first 25% of trading range, last hour = last 25%
    df['morning_volatility'] = (df['high'] - df['low']).rolling(window=2).apply(
        lambda x: x.iloc[0] if len(x) == 2 else np.nan
    )
    df['afternoon_volatility'] = (df['high'] - df['low']).rolling(window=2).apply(
        lambda x: x.iloc[1] if len(x) == 2 else np.nan
    )
    
    # Volatility Expansion Ratio
    df['vol_expansion_ratio'] = np.abs(df['morning_volatility'] / (df['afternoon_volatility'] + 1e-8))
    
    # Calculate Intraday Momentum Acceleration
    df['intraday_momentum'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
    df['prev_intraday_momentum'] = df['intraday_momentum'].shift(1)
    df['momentum_acceleration'] = df['intraday_momentum'] - df['prev_intraday_momentum']
    
    # Combine Volatility Expansion with Momentum Acceleration
    df['vol_momentum_signal'] = df['vol_expansion_ratio'] * df['momentum_acceleration']
    df['vol_momentum_signal'] = df['vol_momentum_signal'] * np.sign(df['momentum_acceleration'])
    
    # Add Volume Flow Confirmation
    # Calculate volume distribution (assuming equal periods for simplicity)
    df['morning_volume'] = df['volume'].rolling(window=2).apply(
        lambda x: x.iloc[0] if len(x) == 2 else np.nan
    )
    df['afternoon_volume'] = df['volume'].rolling(window=2).apply(
        lambda x: x.iloc[1] if len(x) == 2 else np.nan
    )
    
    df['volume_shift_ratio'] = df['afternoon_volume'] / (df['morning_volume'] + 1e-8)
    
    # Combine with Volatility-Momentum Signal
    df['volume_confirmed_signal'] = df['vol_momentum_signal'] * df['volume_shift_ratio']
    
    # Apply volume confirmation threshold
    volume_threshold = df['volume_shift_ratio'].rolling(window=20, min_periods=10).quantile(0.7)
    df['volume_confirmed_signal'] = np.where(
        df['volume_shift_ratio'] > volume_threshold,
        df['volume_confirmed_signal'],
        df['volume_confirmed_signal'] * 0.5  # Reduce signal strength if volume confirmation weak
    )
    
    # Final Factor Calculation
    factor = df['volume_confirmed_signal']
    
    return factor
