import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Efficiency Analysis
    # Volatility Compression: (High_t - Low_t) / (High_{t-1} - Low_{t-1})
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['volatility_compression'] = data['daily_range'] / data['prev_daily_range']
    
    # Price Movement Efficiency: |Close - Open| / (High - Low)
    data['price_movement_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Liquidity Dynamics
    # Volume Velocity: Amount_t / Amount_{t-1}
    data['prev_amount'] = data['amount'].shift(1)
    data['volume_velocity'] = data['amount'] / data['prev_amount']
    
    # Volume Persistence: Count consecutive days with Volume Velocity > 1.2
    data['high_volume_flag'] = (data['volume_velocity'] > 1.2).astype(int)
    data['volume_persistence'] = data['high_volume_flag'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Regime-Adaptive Momentum
    # High Volatility Momentum: |Close - Open| × Volatility Compression
    data['high_vol_momentum'] = abs(data['close'] - data['open']) * data['volatility_compression']
    
    # Volume-Constrained Return: (Close_t - Open_t) / Volume Velocity
    data['volume_constrained_return'] = (data['close'] - data['open']) / data['volume_velocity']
    
    # Composite Alpha Construction
    # Core Signal: Price Movement Efficiency × Volume-Constrained Return
    data['core_signal'] = data['price_movement_efficiency'] * data['volume_constrained_return']
    
    # Final Alpha: Core Signal × Volume Persistence × Volatility Compression
    data['alpha'] = data['core_signal'] * data['volume_persistence'] * data['volatility_compression']
    
    # Return the alpha series
    return data['alpha']
