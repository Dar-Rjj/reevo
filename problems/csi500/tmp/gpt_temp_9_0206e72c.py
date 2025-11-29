import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Reversal-Momentum Convergence with Volatility-Regime Context factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    
    # Intraday Price Behavior Analysis
    # Morning Reversal Component
    data['morning_selling_pressure'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    
    # Afternoon Momentum Component
    data['afternoon_buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_trend'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    
    # Intraday Divergence Detection
    data['intraday_divergence'] = data['morning_selling_pressure'] - data['afternoon_buying_pressure']
    data['divergence_3d_mean'] = data['intraday_divergence'].rolling(window=3, min_periods=1).mean()
    data['divergence_strength'] = data['intraday_divergence'] - data['divergence_3d_mean']
    
    # Volatility-Regime Framework
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Volatility regime
    data['vol_20d_median'] = data['true_range'].rolling(window=20, min_periods=1).median()
    data['vol_regime'] = np.where(data['true_range'] > data['vol_20d_median'], 1, 0)  # 1=high, 0=low
    
    # Compression Analysis
    data['tr_10d_min'] = data['true_range'].rolling(window=10, min_periods=1).min()
    data['tr_10d_max'] = data['true_range'].rolling(window=10, min_periods=1).max()
    data['compression_score'] = (data['true_range'] - data['tr_10d_min']) / (data['tr_10d_max'] - data['tr_10d_min'] + 1e-8)
    data['compression_breakout'] = 1 - abs(data['compression_score'] - 0.5) * 2  # Higher near extremes
    
    # Volume Convergence Assessment
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_10d_avg'] + 1e-8)
    
    # Volume persistence (consecutive days above 20-day average)
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_above_avg'] = (data['volume'] > data['volume_20d_avg']).astype(int)
    data['volume_persistence'] = data['volume_above_avg'].rolling(window=5, min_periods=1).sum()
    
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_spike'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    # Volume-Price Alignment
    data['prev_volume'] = data['volume'].shift(1)
    data['morning_alignment'] = np.sign(data['morning_selling_pressure']) * np.sign(data['volume'] - data['prev_volume'])
    data['afternoon_alignment'] = np.sign(data['afternoon_buying_pressure']) * np.sign(data['volume'] - data['prev_volume'])
    data['overall_alignment'] = (data['morning_alignment'] + data['afternoon_alignment']) / 2
    
    # Liquidity Context
    data['market_depth'] = data['volume'] / ((data['high'] - data['low'] + 1e-8) * data['close'])
    data['implicit_spread'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    
    # Trend Context Integration
    data['primary_trend'] = data['close'].rolling(window=20, min_periods=1).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0, raw=False
    )
    data['secondary_trend'] = data['close'].rolling(window=5, min_periods=1).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0, raw=False
    )
    data['trend_conflict'] = (data['primary_trend'] != data['secondary_trend']).astype(int)
    
    # Trend strength
    data['primary_trend_magnitude'] = (data['close'] - data['close'].shift(20)) / (data['close'].shift(20) + 1e-8)
    data['recent_acceleration'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8) - data['primary_trend_magnitude']
    data['trend_stability'] = data['close'].pct_change().rolling(window=10, min_periods=1).std()
    
    # Signal Generation Framework
    # Core Convergence Signal
    data['base_convergence'] = data['intraday_divergence'] * data['volume_concentration']
    data['vol_adjusted_convergence'] = data['base_convergence'] * (1 + data['compression_score'])
    data['volume_aligned_convergence'] = data['vol_adjusted_convergence'] * (1 + data['overall_alignment'])
    
    # Contextual Enhancement
    data['trend_context'] = data['volume_aligned_convergence'] * (1 + abs(data['primary_trend_magnitude']))
    data['vol_regime_adjusted'] = data['trend_context'] * np.where(data['vol_regime'] == 1, 1.2, 0.8)  # High vol: more sensitive
    data['liquidity_scaled'] = data['vol_regime_adjusted'] * (1 + 1/(data['market_depth'] + 1e-8))
    
    # Signal Refinement
    data['persistence_weighted'] = data['liquidity_scaled'] * (1 + data['volume_persistence'] / 5)
    
    # Regime filtering
    data['regime_filtered'] = np.where(
        data['vol_regime'] == 1,  # High volatility
        data['persistence_weighted'] * data['morning_selling_pressure'],  # Enhanced reversal
        data['persistence_weighted'] * data['afternoon_buying_pressure']   # Enhanced momentum
    )
    
    # Conflict resolution
    data['conflict_resolved'] = data['regime_filtered'] * (1 - data['trend_conflict'] * 0.3)
    
    # Final Factor Construction
    # Primary factor with maturation
    data['primary_factor'] = data['conflict_resolved'].shift(2)  # 2-day maturation
    
    # Smoothing
    data['smoothed_factor'] = data['primary_factor'].rolling(window=5, min_periods=1).mean()
    
    # Persistence check
    data['signal_persistence'] = data['smoothed_factor'].rolling(window=3, min_periods=1).std()
    
    # Risk-adjusted output
    data['volatility_scaling'] = 1 / (data['true_range'].rolling(window=10, min_periods=1).mean() + 1e-8)
    data['regime_weighted'] = data['smoothed_factor'] * data['volatility_scaling']
    data['regime_weighted'] = data['regime_weighted'] * np.where(data['vol_regime'] == 1, 0.8, 1.2)  # Adjust for regime
    
    # Composite score
    data['composite_score'] = data['regime_weighted'] * (1 - data['signal_persistence'])
    
    # Final factor output
    factor = data['composite_score']
    
    return factor
