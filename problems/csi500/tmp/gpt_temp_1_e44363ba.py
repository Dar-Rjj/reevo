import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day close with proper shifting
    data['prev_close'] = data['close'].shift(1)
    
    # 1. Calculate Intraday Range Efficiency
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Range Efficiency Ratio
    data['range_efficiency'] = (data['high'] - data['low']) / data['true_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['range_efficiency'] = data['range_efficiency'].fillna(0)
    
    # 2. Analyze Volume-Price Divergence Pattern
    # Volume Momentum - 3-day rolling median
    data['volume_median_3d'] = data['volume'].rolling(window=3, min_periods=1).median()
    data['volume_momentum'] = data['volume'] / data['volume_median_3d']
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Calculate rolling percentiles for threshold determination (5-day window)
    data['range_eff_pct'] = data['range_efficiency'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['vol_mom_pct'] = data['volume_momentum'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Detect divergence conditions
    data['high_eff_low_vol'] = (data['range_eff_pct'] > 0.7) & (data['vol_mom_pct'] < 0.3)
    data['low_eff_high_vol'] = (data['range_eff_pct'] < 0.3) & (data['vol_mom_pct'] > 0.7)
    
    # Divergence strength calculation
    data['divergence_strength'] = abs(data['range_efficiency'] - data['volume_momentum'])
    
    # Scale by recent volatility (10-day rolling std of returns)
    data['returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['returns'].rolling(window=10, min_periods=1).std()
    data['volatility_10d'] = data['volatility_10d'].replace(0, np.nan).fillna(method='ffill').fillna(0.01)
    data['scaled_divergence'] = data['divergence_strength'] / data['volatility_10d']
    
    # 3. Incorporate Intraday Momentum
    data['price_movement'] = (data['close'] - data['open']) / data['open']
    data['momentum_enhanced_efficiency'] = data['range_efficiency'] * data['price_movement']
    
    # 4. Apply Volatility Scaling
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=1).std()
    data['volatility_20d'] = data['volatility_20d'].replace(0, np.nan).fillna(method='ffill').fillna(0.01)
    data['scaled_momentum_efficiency'] = data['momentum_enhanced_efficiency'] / data['volatility_20d']
    
    # 5. Generate Composite Alpha Factor
    # Combine divergence and momentum signals with conditional logic
    data['composite_signal'] = data['scaled_divergence'] * data['scaled_momentum_efficiency']
    
    # Apply sign based on divergence type
    data['divergence_sign'] = 0
    data.loc[data['high_eff_low_vol'], 'divergence_sign'] = 1
    data.loc[data['low_eff_high_vol'], 'divergence_sign'] = -1
    
    data['signed_composite'] = data['composite_signal'] * data['divergence_sign']
    
    # Incorporate Volume Confirmation
    data['volume_avg_20d'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_avg_20d']
    data['volume_confirmed_signal'] = data['signed_composite'] * data['volume_ratio']
    
    # Apply Price Level Adjustment
    data['final_factor'] = data['volume_confirmed_signal'] / data['close']
    
    # Clean up intermediate columns
    result = data['final_factor'].copy()
    
    return result
