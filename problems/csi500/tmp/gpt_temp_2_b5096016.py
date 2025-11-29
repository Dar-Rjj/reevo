import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Volume-Price Elasticity Regime Detection
    Generates alpha factors based on volume-price relationship dynamics
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and ranges
    data['returns'] = data['close'].pct_change()
    data['range'] = (data['high'] - data['low']) / data['close']
    data['open_to_close'] = (data['close'] - data['open']) / data['open']
    
    # Multi-Timeframe Elasticity Dynamics
    # Immediate Price Impact Analysis
    data['intraday_volume_shock'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    data['price_impact_immediate'] = data['open_to_close'] / (data['intraday_volume_shock'] + 1e-8)
    
    # Opening vs Closing Impact Differences
    data['opening_volume_intensity'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x.iloc[0] / x.mean() if len(x) == 5 else np.nan
    )
    data['closing_volume_intensity'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x.iloc[-1] / x.mean() if len(x) == 5 else np.nan
    )
    
    # Delayed Elasticity Effects
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_acceleration'] = data['volume_ma_5'] / data['volume_ma_10'] - 1
    
    # Multi-day impact persistence
    data['cumulative_volume_5d'] = data['volume'].rolling(window=5, min_periods=3).sum()
    data['cumulative_return_5d'] = data['returns'].rolling(window=5, min_periods=3).apply(
        lambda x: (1 + x).prod() - 1
    )
    data['elasticity_persistence'] = data['cumulative_return_5d'] / (data['cumulative_volume_5d'] / data['volume_ma_5'] + 1e-8)
    
    # Volume Distribution Asymmetry
    # Session-based volume concentration
    data['volume_concentration'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
    )
    
    # Volume flow persistence
    data['volume_trend_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['volume_persistence'] = data['volume_trend_5d'].rolling(window=5, min_periods=3).mean()
    
    # Range-Volume Efficiency Alignment
    # Volume-weighted range performance
    data['range_efficiency'] = data['range'] / (data['volume'] / data['volume_ma_5'] + 1e-8)
    data['volume_range_utilization'] = data['range'] * data['volume'] / (data['volume_ma_5'] * data['range'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Multi-timeframe volume efficiency
    data['short_term_efficiency'] = data['range'] / (data['volume'] / data['volume_ma_5'] + 1e-8)
    data['medium_term_efficiency'] = data['range'].rolling(window=5, min_periods=3).mean() / (data['volume_ma_5'] / data['volume_ma_10'] + 1e-8)
    
    # Elasticity-Regime Classification
    # High elasticity regime characteristics
    data['strong_price_response'] = (data['price_impact_immediate'].abs() > 
                                   data['price_impact_immediate'].rolling(window=20, min_periods=10).quantile(0.7))
    data['efficient_range_utilization'] = (data['range_efficiency'] > 
                                         data['range_efficiency'].rolling(window=20, min_periods=10).quantile(0.6))
    data['concentrated_volume'] = (data['volume_concentration'] > 
                                 data['volume_concentration'].rolling(window=20, min_periods=10).quantile(0.7))
    
    # Low elasticity regime features
    data['weak_price_impact'] = (data['price_impact_immediate'].abs() < 
                               data['price_impact_immediate'].rolling(window=20, min_periods=10).quantile(0.3))
    data['inefficient_alignment'] = (data['range_efficiency'] < 
                                   data['range_efficiency'].rolling(window=20, min_periods=10).quantile(0.4))
    
    # Regime-based scoring
    high_elasticity_score = (data['strong_price_response'].astype(int) + 
                           data['efficient_range_utilization'].astype(int) + 
                           data['concentrated_volume'].astype(int))
    
    low_elasticity_score = (data['weak_price_impact'].astype(int) + 
                          data['inefficient_alignment'].astype(int))
    
    # Transition signals
    data['volume_accel_signal'] = (data['volume_acceleration'] > 
                                 data['volume_acceleration'].rolling(window=20, min_periods=10).quantile(0.7))
    data['elasticity_change'] = data['elasticity_persistence'].pct_change(3)
    
    # Generate final regime-based predictive signals
    regime_signal = np.where(
        high_elasticity_score >= 2,
        # High elasticity regime: focus on volume confirmation
        data['price_impact_immediate'] * data['volume_range_utilization'],
        np.where(
            low_elasticity_score >= 1,
            # Low elasticity regime: emphasize range efficiency
            data['range_efficiency'] * data['medium_term_efficiency'],
            # Transition period: volume acceleration signals
            data['volume_accel_signal'].astype(int) * data['elasticity_change']
        )
    )
    
    # Multi-timeframe regime consistency
    data['regime_consistency'] = high_elasticity_score.rolling(window=5, min_periods=3).mean()
    
    # Final signal with confidence weighting
    final_signal = regime_signal * (1 + data['regime_consistency'] / 3)
    
    # Return as pandas Series
    return pd.Series(final_signal, index=data.index, name='volume_price_elasticity_regime')
