import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor based on microstructure regime analysis
    Combines price efficiency, volume concentration, and range exhaustion dynamics
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Efficiency Patterns
    # Opening efficiency: gap vs intraday range ratio
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    daily_range = (data['high'] - data['low']) / data['close']
    opening_efficiency = np.abs(opening_gap) / (daily_range + 1e-8)
    
    # Closing efficiency: last hour momentum persistence
    close_to_open = (data['close'] - data['open']) / data['open']
    mid_day_price = (data['high'] + data['low']) / 2
    closing_efficiency = (data['close'] - mid_day_price) / (data['high'] - data['low'] + 1e-8)
    
    # High-low range efficiency vs close-to-close volatility
    close_volatility = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    range_efficiency = daily_range / (close_volatility + 1e-8)
    
    # Price reversal frequency (intraday oscillation)
    intraday_oscillation = ((data['high'] - data['open']) + (data['open'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    reversal_frequency = intraday_oscillation.rolling(window=3, min_periods=2).std()
    
    # 2. Volume Concentration Asymmetry
    # First hour volume concentration (proxy using opening volume intensity)
    volume_ma = data['volume'].rolling(window=5, min_periods=3).mean()
    opening_volume_ratio = data['volume'] / (volume_ma + 1e-8)
    
    # Volume clustering persistence
    volume_zscore = (data['volume'] - data['volume'].rolling(window=10, min_periods=5).mean()) / (data['volume'].rolling(window=10, min_periods=5).std() + 1e-8)
    volume_clustering = volume_zscore.rolling(window=3, min_periods=2).sum()
    
    # Volume-return clustering correlation
    returns = data['close'].pct_change()
    volume_return_corr = data['volume'].rolling(window=5, min_periods=3).corr(returns)
    
    # 3. Price Range Exhaustion Dynamics
    # Range utilization: actual vs potential range
    prev_close = data['close'].shift(1)
    potential_range = np.maximum(data['high'] - prev_close, prev_close - data['low'])
    range_utilization = (data['high'] - data['low']) / (potential_range + 1e-8)
    
    # Range compression/expansion cycles
    range_ma = daily_range.rolling(window=5, min_periods=3).mean()
    range_regime = daily_range / (range_ma + 1e-8)
    
    # Range persistence and boundary testing
    high_retest = (data['high'] - data['high'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    low_retest = (data['low'].shift(1) - data['low']) / (data['high'] - data['low'] + 1e-8)
    boundary_testing = np.abs(high_retest) + np.abs(low_retest)
    
    # 4. Microstructure Regime Integration
    # Efficiency-concentration alignment
    efficiency_concentration = opening_efficiency * opening_volume_ratio
    
    # Range-efficiency interaction
    range_efficiency_interaction = range_efficiency * closing_efficiency
    
    # Volume concentration during price discovery
    volume_price_discovery = volume_clustering * reversal_frequency
    
    # Composite microstructure alpha
    # Weight components by their persistence and information content
    alpha_components = pd.DataFrame({
        'efficiency_align': -efficiency_concentration,  # Negative: rapid efficiency restoration
        'range_efficiency': range_efficiency_interaction,
        'volume_discovery': volume_price_discovery,
        'range_util': range_utilization,
        'boundary_test': -boundary_testing  # Negative: excessive boundary testing suggests exhaustion
    })
    
    # Normalize components and combine with regime-appropriate weights
    normalized_components = alpha_components.apply(lambda x: (x - x.rolling(window=20, min_periods=10).mean()) / (x.rolling(window=20, min_periods=10).std() + 1e-8))
    
    # Final composite score with regime adjustment
    composite_alpha = (
        0.3 * normalized_components['efficiency_align'] +
        0.25 * normalized_components['range_efficiency'] +
        0.25 * normalized_components['volume_discovery'] +
        0.1 * normalized_components['range_util'] +
        0.1 * normalized_components['boundary_test']
    )
    
    # Apply cross-sectional ranking
    cross_sectional_alpha = composite_alpha.groupby(composite_alpha.index).transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    return cross_sectional_alpha
