import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price changes for different periods
    data['price_change_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_change_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_change_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_change_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Compute momentum divergence
    data['momentum_div_short'] = data['price_change_3d'] - data['price_change_5d']
    data['momentum_div_long'] = data['price_change_10d'] - data['price_change_15d']
    data['momentum_divergence'] = data['momentum_div_short'] + data['momentum_div_long']
    
    # Analyze volume-price relationship
    # Price movement per unit volume (absolute price change divided by volume)
    data['price_movement'] = (data['high'] - data['low']) / data['close']
    data['price_per_volume'] = data['price_movement'] / (data['volume'] + 1e-8)  # Avoid division by zero
    
    # Calculate VWAP (Volume Weighted Average Price)
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=5, min_periods=1).sum() / data['volume'].rolling(window=5, min_periods=1).sum()
    data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
    
    # Combine volume-price signals
    data['volume_price_signal'] = data['price_per_volume'] * data['vwap_deviation']
    
    # Generate combined factor
    data['combined_factor'] = data['momentum_divergence'] * data['volume_price_signal']
    
    # Apply cross-sectional ranking
    def cross_sectional_rank(group):
        return group.rank(pct=True)
    
    data['factor_rank'] = data.groupby(data.index)['combined_factor'].transform(cross_sectional_rank)
    
    # Return the final factor values
    return data['factor_rank']
