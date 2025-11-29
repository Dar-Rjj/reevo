import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns from Close prices
    data['returns'] = data['close'].pct_change()
    
    # Calculate rolling volatility (10-day standard deviation)
    data['volatility'] = data['returns'].rolling(window=10).std()
    
    # Calculate volatility percentiles for regime classification
    vol_60th = data['volatility'].rolling(window=50, min_periods=20).quantile(0.6)
    vol_40th = data['volatility'].rolling(window=50, min_periods=20).quantile(0.4)
    
    # Classify volatility regimes
    high_vol_regime = data['volatility'] > vol_60th
    low_vol_regime = data['volatility'] < vol_40th
    
    # Calculate turnover rate (using amount as proxy for shares outstanding)
    # Turnover = volume / (amount / close) = (volume * close) / amount
    data['turnover'] = (data['volume'] * data['close']) / data['amount']
    
    # Calculate turnover rate of change
    data['turnover_roc_3d'] = data['turnover'].pct_change(periods=3)
    data['turnover_roc_8d'] = data['turnover'].pct_change(periods=8)
    
    # Calculate liquidity acceleration (3-day ROC minus 8-day ROC)
    data['liquidity_acceleration'] = data['turnover_roc_3d'] - data['turnover_roc_8d']
    
    # Calculate smoothed liquidity acceleration for low volatility regime
    data['smoothed_acceleration'] = data['liquidity_acceleration'].rolling(window=5).mean()
    
    # Calculate momentum decay weights for high volatility regime
    decay_weights = np.exp(-np.arange(5) / 3)  # 3-day half-life
    decay_weights = decay_weights / decay_weights.sum()
    
    # Apply momentum decay weighting for high volatility regime
    data['weighted_acceleration'] = 0.0
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days for the window
            window_data = data['liquidity_acceleration'].iloc[i-4:i+1]
            data.iloc[i, data.columns.get_loc('weighted_acceleration')] = (window_data * decay_weights).sum()
    
    # Calculate mean reversion enhancement for low volatility regime
    data['acceleration_ma_15d'] = data['liquidity_acceleration'].rolling(window=15).mean()
    data['deviation_from_ma'] = data['liquidity_acceleration'] - data['acceleration_ma_15d']
    data['mean_reversion_enhanced'] = data['smoothed_acceleration'] * data['deviation_from_ma']
    
    # Construct regime-adaptive factor
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if high_vol_regime.iloc[i]:
            # High volatility regime: use weighted acceleration
            factor_values.iloc[i] = data['weighted_acceleration'].iloc[i]
        elif low_vol_regime.iloc[i]:
            # Low volatility regime: use mean reversion enhanced acceleration
            factor_values.iloc[i] = data['mean_reversion_enhanced'].iloc[i]
        else:
            # Normal regime: use raw liquidity acceleration
            factor_values.iloc[i] = data['liquidity_acceleration'].iloc[i]
    
    return factor_values
