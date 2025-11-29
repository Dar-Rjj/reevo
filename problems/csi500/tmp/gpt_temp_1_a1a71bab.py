import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize factor series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['daily_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['transaction_size'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['daily_range'] = df['high'] - df['low']
    df['prev_range'] = df['daily_range'].shift(1)
    df['volatility_regime'] = df['daily_range'] / df['prev_range'].replace(0, np.nan)
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Intraday Efficiency Momentum
    df['efficiency_3d_ma'] = df['daily_efficiency'].rolling(window=3, min_periods=1).mean()
    efficiency_momentum = df['daily_efficiency'] - df['efficiency_3d_ma']
    
    # Transaction Intensity Analysis
    df['prev_transaction_size'] = df['transaction_size'].shift(1)
    transaction_momentum = df['transaction_size'] / df['prev_transaction_size'].replace(0, np.nan) - 1
    
    # Volatility Transition Efficiency
    volatility_efficiency = df['daily_efficiency'] * df['volatility_regime']
    
    # Multi-Scale Pressure Dynamics
    pressure_asymmetry = df['daily_efficiency']
    df['efficiency_gradient'] = df['daily_efficiency'].diff()
    df['efficiency_persistence'] = df['daily_efficiency'].rolling(window=3, min_periods=1).std()
    gradient_persistence_alignment = df['efficiency_gradient'] / df['efficiency_persistence'].replace(0, np.nan)
    
    # Liquidity Barrier Assessment
    df['volume_ma_5d'] = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_concentration = df['volume'] / df['volume_ma_5d'].replace(0, np.nan)
    df['amount_ma_5d'] = df['amount'].rolling(window=5, min_periods=1).mean()
    amount_surge = df['amount'] / df['amount_ma_5d'].replace(0, np.nan)
    liquidity_barrier = volume_concentration * amount_surge
    
    # Temporal Alignment Momentum
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    price_volume_correlation = df['price_change'].rolling(window=5, min_periods=1).corr(df['volume_change'])
    alignment_quality = price_volume_correlation * np.sign(df['price_change'])
    
    # Microstructure Gap Quality
    df['gap_fill_ratio'] = (df['close'] - df['open']) / df['opening_gap'].replace(0, np.nan)
    df['gap_volume_ratio'] = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    gap_quality = df['gap_fill_ratio'] * df['gap_volume_ratio']
    
    # Range Expansion Quality
    df['range_ma_5d'] = df['daily_range'].rolling(window=5, min_periods=1).mean()
    range_expansion = df['daily_range'] / df['range_ma_5d'].replace(0, np.nan)
    df['volume_ma_3d'] = df['volume'].rolling(window=3, min_periods=1).mean()
    volume_confirmation = df['volume'] / df['volume_ma_3d'].replace(0, np.nan)
    range_quality = range_expansion * volume_confirmation
    
    # Combine all components with equal weights
    components = [
        efficiency_momentum,
        transaction_momentum,
        volatility_efficiency,
        gradient_persistence_alignment,
        liquidity_barrier,
        alignment_quality,
        gap_quality,
        range_quality
    ]
    
    # Standardize each component and combine
    for i, component in enumerate(components):
        if not component.isnull().all():
            # Cross-sectional standardization
            component_standardized = component.groupby(component.index).transform(lambda x: (x - x.mean()) / x.std())
            if i == 0:
                combined_factor = component_standardized
            else:
                combined_factor = combined_factor.add(component_standardized, fill_value=0)
    
    factor = combined_factor
    
    return factor
