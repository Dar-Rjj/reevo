import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate volume-weighted price
    data['vw_price'] = (data['close'] * data['volume']) / data['volume']
    
    # Calculate first derivative (rate of change) for volume-weighted price
    data['vw_roc'] = data['vw_price'].diff() / data['vw_price'].shift(1)
    
    # Calculate second derivative (acceleration) for volume-weighted price
    data['vw_accel'] = data['vw_roc'].diff() / data['vw_roc'].shift(1)
    
    # Calculate first derivative (rate of change) for pure price
    data['price_roc'] = data['close'].diff() / data['close'].shift(1)
    
    # Calculate second derivative (acceleration) for pure price
    data['price_accel'] = data['price_roc'].diff() / data['price_roc'].shift(1)
    
    # Calculate absolute divergence between volume-weighted and pure acceleration
    data['accel_divergence'] = data['vw_accel'] - data['price_accel']
    
    # Calculate rolling correlation to identify leading signals
    # Use 5-day window to capture short-term leading relationships
    rolling_corr = []
    for i in range(len(data)):
        if i < 5:
            rolling_corr.append(np.nan)
        else:
            window = data.iloc[i-5:i]
            # Check if we have enough non-NaN values
            valid_vw = window['vw_accel'].dropna()
            valid_price = window['price_accel'].dropna()
            if len(valid_vw) >= 3 and len(valid_price) >= 3:
                # Use cross-correlation to detect leading relationship
                corr = valid_vw.corr(valid_price.shift(-1).dropna())  # vw leads price
                rolling_corr.append(corr)
            else:
                rolling_corr.append(np.nan)
    
    data['vw_leading_corr'] = rolling_corr
    
    # Generate factor: positive when volume-weighted acceleration leads pure price acceleration
    # and there's positive divergence
    factor = np.where(
        (data['vw_leading_corr'] > 0.3) &  # Strong leading relationship
        (data['accel_divergence'] > 0) &    # Volume-weighted acceleration is stronger
        (data['vw_accel'] > 0),             # Positive momentum in volume-weighted acceleration
        data['accel_divergence'] * data['vw_leading_corr'],  # Amplify by correlation strength
        0
    )
    
    # Create output series
    factor_series = pd.Series(factor, index=data.index)
    
    return factor_series
