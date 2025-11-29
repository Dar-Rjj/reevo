import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Directional Momentum Components
    # Bullish vs Bearish Intraday Efficiency
    data['directional_bias'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # High-Low Range Divergence
    data['high_momentum'] = data['high'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['low_momentum'] = data['low'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['range_divergence'] = data['high_momentum'] - data['low_momentum']
    
    # 2. Volume Efficiency Patterns
    # Volume-Weighted Price Efficiency
    data['vw_price_efficiency'] = (data['close'] * data['volume']) / (data['amount'] + 1e-8)
    
    # Volume Concentration Asymmetry
    # Calculate volume skewness during up vs down days
    data['price_change'] = data['close'].pct_change()
    data['up_day_volume'] = data['volume'].where(data['price_change'] > 0)
    data['down_day_volume'] = data['volume'].where(data['price_change'] < 0)
    
    # Rolling volume skewness (5-day window)
    def rolling_skewness(x):
        if len(x) < 3:
            return 0
        mean_x = np.mean(x)
        std_x = np.std(x) + 1e-8
        return np.mean(((x - mean_x) / std_x) ** 3)
    
    data['up_volume_skew'] = data['up_day_volume'].rolling(window=5).apply(rolling_skewness, raw=True)
    data['down_volume_skew'] = data['down_day_volume'].rolling(window=5).apply(rolling_skewness, raw=True)
    data['volume_concentration_asymmetry'] = data['up_volume_skew'] - data['down_volume_skew']
    
    # 3. Asymmetric Regime Signals
    # Convergence Factor - Directional bias alignment with volume efficiency
    data['convergence_factor'] = data['directional_bias'] * data['vw_price_efficiency']
    
    # Divergence Signal - Price-volume asymmetry indicating regime shifts
    # Calculate correlation between price changes and volume changes
    data['volume_change'] = data['volume'].pct_change()
    
    def rolling_corr(x):
        if len(x) < 3:
            return 0
        price_changes = x[:, 0]
        volume_changes = x[:, 1]
        valid_mask = (~np.isnan(price_changes)) & (~np.isnan(volume_changes))
        if np.sum(valid_mask) < 3:
            return 0
        return np.corrcoef(price_changes[valid_mask], volume_changes[valid_mask])[0, 1]
    
    combined_data = np.column_stack([data['price_change'].values, data['volume_change'].values])
    data['price_volume_corr'] = pd.Series(
        [rolling_corr(combined_data[max(0, i-4):i+1]) for i in range(len(data))],
        index=data.index
    )
    
    # Final factor: Combine all components with appropriate weights
    # Normalize components to similar scales
    components = ['directional_bias', 'range_divergence', 'vw_price_efficiency', 
                 'volume_concentration_asymmetry', 'convergence_factor', 'price_volume_corr']
    
    # Remove any infinite values
    for col in components:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    
    # Z-score normalization (using rolling 20-day mean and std to avoid lookahead bias)
    normalized_components = []
    for col in components:
        rolling_mean = data[col].rolling(window=20, min_periods=10).mean()
        rolling_std = data[col].rolling(window=20, min_periods=10).std() + 1e-8
        normalized = (data[col] - rolling_mean) / rolling_std
        normalized_components.append(normalized)
    
    # Combine components with weights reflecting the thought tree structure
    factor = (
        0.25 * normalized_components[0] +  # directional_bias
        0.20 * normalized_components[1] +  # range_divergence
        0.15 * normalized_components[2] +  # vw_price_efficiency
        0.15 * normalized_components[3] +  # volume_concentration_asymmetry
        0.15 * normalized_components[4] +  # convergence_factor
        0.10 * normalized_components[5]    # price_volume_corr
    )
    
    return factor
