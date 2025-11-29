import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Volume Interaction Alpha Factor
    Combines multiple microstructural signals to capture volatility absorption,
    flow concentration, and regime transition patterns.
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Volatility Absorption Analysis
    # Calculate average volume over 20 days
    data['avg_volume_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    # Volatility absorption capacity
    data['vol_absorption'] = (data['high'] - data['low']) / (data['volume'] / data['avg_volume_20'].replace(0, 1))
    data['vol_absorption'] = data['vol_absorption'].replace([np.inf, -np.inf], np.nan)
    
    # Absorption regime classification
    absorption_high = data['vol_absorption'].rolling(window=10, min_periods=5).quantile(0.7)
    absorption_low = data['vol_absorption'].rolling(window=10, min_periods=5).quantile(0.3)
    volatility_high = (data['high'] - data['low']).rolling(window=10, min_periods=5).quantile(0.7)
    volatility_low = (data['high'] - data['low']).rolling(window=10, min_periods=5).quantile(0.3)
    
    # Speculative pressure (high vol, low absorption)
    data['spec_pressure'] = ((data['high'] - data['low']) > volatility_high) & (data['vol_absorption'] < absorption_low)
    # Accumulation quality (low vol, high absorption)
    data['accum_quality'] = ((data['high'] - data['low']) < volatility_low) & (data['vol_absorption'] > absorption_high)
    
    # 2. Price-Range Expansion Dynamics
    data['prev_range'] = (data['high'] - data['low']).shift(1)
    data['range_expansion'] = (data['high'] - data['low']) / data['prev_range'].replace(0, 1)
    data['range_expansion'] = data['range_expansion'].replace([np.inf, -np.inf], np.nan)
    
    # Expansion-volume alignment
    vol_rank = data['volume'].rolling(window=20, min_periods=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    exp_rank = data['range_expansion'].rolling(window=20, min_periods=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    data['exp_vol_alignment'] = exp_rank - vol_rank
    
    # 3. Microstructural Flow Concentration
    # Volume clustering using rolling standard deviation
    data['volume_std_5'] = data['volume'].rolling(window=5, min_periods=3).std()
    data['volume_mean_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['flow_concentration'] = data['volume_std_5'] / data['volume_mean_5'].replace(0, 1)
    data['flow_concentration'] = data['flow_concentration'].replace([np.inf, -np.inf], np.nan)
    
    # Concentration persistence
    data['conc_persistence'] = data['flow_concentration'].rolling(window=5, min_periods=3).std()
    
    # 4. Volatility Regime Transition Detection
    data['volatility_20'] = (data['high'] - data['low']).rolling(window=20, min_periods=10).std()
    data['vol_regime_change'] = data['volatility_20'].pct_change(5)
    
    # Volume-volatility relationship
    vol_corr_window = 10
    data['vol_vol_correlation'] = data['volume'].rolling(window=vol_corr_window).corr(data['high'] - data['low'])
    
    # 5. Price-Volume Divergence Architecture
    price_momentum = data['close'].pct_change(5)
    volume_momentum = data['volume'].pct_change(5)
    data['price_volume_divergence'] = price_momentum - volume_momentum
    
    # Divergence persistence
    data['divergence_persistence'] = data['price_volume_divergence'].rolling(window=5, min_periods=3).std()
    
    # 6. Session Boundary Flow Analysis
    data['prev_close'] = data['close'].shift(1)
    data['opening_flow'] = (data['open'] - data['prev_close']) * data['volume']
    data['closing_flow'] = (data['close'] - data['open']) * data['volume']
    data['boundary_flow_ratio'] = data['opening_flow'] / (abs(data['closing_flow']) + 1e-10)
    
    # 7. Volatility Compression-Expansion Cycles
    data['range_5'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['range_20'] = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    data['compression_ratio'] = data['range_5'] / data['range_20'].replace(0, 1)
    
    # Cycle-volume relationship
    data['cycle_volume_alignment'] = data['compression_ratio'] * data['volume'] / data['avg_volume_20'].replace(0, 1)
    
    # 8. Flow Imbalance Momentum
    # Estimate buy-sell pressure using close position in daily range
    data['range_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1)
    data['imbalance_strength'] = (data['range_position'] - 0.5) * data['volume']
    
    # Imbalance persistence
    data['imbalance_persistence'] = data['imbalance_strength'].rolling(window=5, min_periods=3).mean()
    
    # Combine signals with appropriate weights
    factors = []
    
    # Volatility absorption signal (negative for high speculative pressure)
    absorption_signal = -data['spec_pressure'].astype(float) + data['accum_quality'].astype(float)
    factors.append(absorption_signal.rolling(window=5, min_periods=3).mean())
    
    # Range expansion with volume alignment
    expansion_signal = data['range_expansion'] * data['exp_vol_alignment']
    factors.append(expansion_signal.rolling(window=5, min_periods=3).mean())
    
    # Flow concentration quality (negative for high concentration breakdown)
    concentration_signal = -data['flow_concentration'] / (data['conc_persistence'] + 1e-10)
    factors.append(concentration_signal.rolling(window=5, min_periods=3).mean())
    
    # Regime transition momentum
    regime_signal = data['vol_regime_change'] * data['vol_vol_correlation']
    factors.append(regime_signal.rolling(window=5, min_periods=3).mean())
    
    # Price-volume divergence strength
    divergence_signal = data['price_volume_divergence'] / (data['divergence_persistence'] + 1e-10)
    factors.append(divergence_signal.rolling(window=5, min_periods=3).mean())
    
    # Boundary flow continuity
    boundary_signal = data['boundary_flow_ratio'] * np.sign(data['closing_flow'])
    factors.append(boundary_signal.rolling(window=5, min_periods=3).mean())
    
    # Compression-expansion cycle quality
    cycle_signal = data['compression_ratio'] * data['cycle_volume_alignment']
    factors.append(cycle_signal.rolling(window=5, min_periods=3).mean())
    
    # Flow imbalance momentum
    imbalance_signal = data['imbalance_strength'] / (abs(data['imbalance_persistence']) + 1e-10)
    factors.append(imbalance_signal.rolling(window=5, min_periods=3).mean())
    
    # Combine all factors with equal weighting
    combined_factor = pd.concat(factors, axis=1).mean(axis=1)
    
    # Final normalization
    final_factor = (combined_factor - combined_factor.rolling(window=20, min_periods=10).mean()) / combined_factor.rolling(window=20, min_periods=10).std()
    
    return final_factor
