import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Price-Volume Efficiency Signal
    # Calculate Intraday Efficiency Ratio
    price_range_efficiency = (data['high'] - data['low']) / (data['close'] - data['open'] + 1e-12)
    volume_distribution_skew = (data['volume'] * (data['high'] - data['low']) / (data['high'] - data['low'] + 1e-12)) / data['volume']
    efficiency_ratio = price_range_efficiency * volume_distribution_skew
    
    # Calculate Volume Persistence
    volume_momentum = data['volume'] / data['volume'].shift(2).replace(0, 1e-12)
    volume_stability = data['volume'] / data['volume'].rolling(window=7, min_periods=1).median().replace(0, 1e-12)
    volume_persistence = volume_momentum * volume_stability
    
    # Generate Efficiency Signal
    efficiency_signal = efficiency_ratio * volume_persistence
    
    # Calculate Volatility Regime Components
    # Price Volatility Regime
    price_volatility = data['close'].pct_change().rolling(window=8, min_periods=1).std()
    volatility_momentum = price_volatility / price_volatility.shift(4).replace(0, 1e-12)
    price_volatility_regime = np.log1p(np.abs(volatility_momentum)) * np.sign(volatility_momentum)
    
    # Volume Volatility Regime
    volume_changes = data['volume'].pct_change()
    volume_volatility = volume_changes.rolling(window=5, min_periods=1).std()
    volume_volatility_ratio = volume_volatility / volume_volatility.rolling(window=10, min_periods=1).mean().replace(0, 1e-12)
    
    # Generate Regime Signal
    regime_signal = price_volatility_regime * volume_volatility_ratio
    
    # Construct Multi-horizon Signal
    # Immediate Component
    intraday_trend_filter = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-12)
    immediate_component = efficiency_signal * intraday_trend_filter
    
    # Momentum Component
    efficiency_change = efficiency_signal - efficiency_signal.shift(3)
    volume_acceleration = data['volume'] / data['volume'].shift(3).replace(0, 1e-12)
    momentum_component = efficiency_change * volume_acceleration
    
    # Combine Horizons
    weighted_product = (np.abs(immediate_component) ** 0.6) * (np.abs(momentum_component) ** 0.4) * np.sign(immediate_component * momentum_component)
    multi_horizon_signal = weighted_product * regime_signal
    
    # Apply Volume-Volatility Confirmation
    # Volume-Volatility Divergence
    volume_volatility_corr = data['volume'].rolling(window=5, min_periods=1).corr(price_volatility)
    avg_correlation = volume_volatility_corr.rolling(window=10, min_periods=1).mean()
    volume_volatility_divergence = volume_volatility_corr - avg_correlation
    
    # Volume Spike Persistence
    volume_median = data['volume'].rolling(window=10, min_periods=1).median()
    volume_spikes = (data['volume'] > 1.5 * volume_median).astype(int)
    current_spike_count = volume_spikes.rolling(window=1, min_periods=1).sum()
    five_day_spike_count = volume_spikes.rolling(window=5, min_periods=1).sum()
    persistence_ratio = current_spike_count / five_day_spike_count.replace(0, 1e-12)
    
    # Generate Final Factor
    confirmation_signal = volume_volatility_divergence * persistence_ratio
    combined_factor = multi_horizon_signal * confirmation_signal
    
    # Apply volatility-based threshold
    price_volatility_20d = data['close'].pct_change().rolling(window=20, min_periods=1).std()
    volatility_quantile = price_volatility_20d.rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Final factor with volatility threshold
    final_factor = combined_factor * (volatility_quantile > 0.3)
    
    return final_factor
