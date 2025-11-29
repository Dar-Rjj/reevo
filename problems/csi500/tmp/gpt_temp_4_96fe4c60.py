import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate cross-sectional alpha factors using price, volume, and amount data
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Based Momentum Factors
    # Intraday Range Persistence
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_pct'] = data['daily_range'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - np.percentile(x[:-1], 30)) / (np.percentile(x[:-1], 70) - np.percentile(x[:-1], 30)) 
        if len(x) >= 3 and (np.percentile(x[:-1], 70) - np.percentile(x[:-1], 30)) != 0 else 0
    )
    
    # Opening Gap Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['open_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_accel'] = data['volume'] / data['volume_5d_avg']
    data['gap_efficiency'] = data['open_gap'] * np.log1p(data['volume_accel'])
    
    # Volume-Price Interaction Factors
    # Amount-Based Price Impact
    data['price_change'] = data['close'] - data['open']
    data['amount_price_impact'] = data['price_change'] / (data['amount'] + 1e-8)
    data['impact_20d_median'] = data['amount_price_impact'].rolling(window=20, min_periods=10).median()
    data['impact_deviation'] = (data['amount_price_impact'] - data['impact_20d_median']) / (data['impact_20d_median'].abs() + 1e-8)
    
    # Volume-Weighted Elasticity
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['price_elasticity'] = data['price_change'] / (data['volume_change'].abs() + 1e-8)
    data['elasticity_10d_std'] = data['price_elasticity'].rolling(window=10, min_periods=5).std()
    data['elasticity_anomaly'] = (data['price_elasticity'] - data['price_elasticity'].rolling(window=10, min_periods=5).mean()) / (data['elasticity_10d_std'] + 1e-8)
    
    # Price Path Efficiency Factors
    # Intraday Movement Efficiency
    data['optimal_path'] = np.abs(data['close'] - data['open'])
    data['actual_path'] = np.abs(data['high'] - data['low'])
    data['movement_efficiency'] = data['optimal_path'] / (data['actual_path'] + 1e-8)
    
    # Multi-Timeframe Volatility Convergence
    data['range_1d'] = (data['high'] - data['low']) / data['close']
    data['range_3d'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['range_3d'] = data['range_3d'] / data['close']
    data['range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['range_5d'] = data['range_5d'] / data['close']
    data['volatility_ratio'] = (data['range_1d'] + 1e-8) / (data['range_5d'] + 1e-8)
    
    # Reversal and Regime Factors
    # Close-to-Open Reversal Strength
    data['overnight_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_extreme'] = np.maximum(np.abs(data['high'] - data['open']), np.abs(data['low'] - data['open'])) / data['open']
    data['reversal_magnitude'] = -data['overnight_return'] * data['intraday_extreme']
    
    # Volatility Compression Detection
    data['range_10d_avg'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    data['range_10d_std'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    data['vol_compression'] = (data['daily_range'] - data['range_10d_avg']) / (data['range_10d_std'] + 1e-8)
    
    # Combine factors with weights
    factors = pd.DataFrame(index=data.index)
    factors['factor_range_persistence'] = data['range_5d_pct']
    factors['factor_gap_efficiency'] = data['gap_efficiency']
    factors['factor_impact_deviation'] = data['impact_deviation']
    factors['factor_elasticity_anomaly'] = data['elasticity_anomaly']
    factors['factor_movement_efficiency'] = data['movement_efficiency']
    factors['factor_volatility_ratio'] = data['volatility_ratio']
    factors['factor_reversal_magnitude'] = data['reversal_magnitude']
    factors['factor_vol_compression'] = data['vol_compression']
    
    # Standardize each factor
    for col in factors.columns:
        factors[col] = (factors[col] - factors[col].mean()) / (factors[col].std() + 1e-8)
    
    # Final composite factor (equal weighted)
    final_factor = factors.mean(axis=1)
    
    return final_factor
