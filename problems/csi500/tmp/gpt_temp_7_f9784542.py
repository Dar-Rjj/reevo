import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Momentum Efficiency Asymmetry Factor
    # Directional Momentum Component
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Opening Momentum
    data['opening_momentum'] = (data['prev_close'] - data['open']) / (data['prev_high'] - data['prev_low'] + 1e-8)
    
    # Intraday Efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum Asymmetry
    data['momentum_asymmetry'] = data['opening_momentum'] * data['intraday_efficiency']
    
    # Volume-Confirmed Momentum
    # Volume Acceleration
    data['volume_acceleration'] = data['volume'] / (data['prev_volume'] + 1e-8) - 1
    
    # Volume Distribution
    data['volume_distribution'] = (data['high'] - data['close']) * data['volume']
    
    # Volume Momentum
    data['volume_momentum'] = data['volume_acceleration'] * data['volume_distribution']
    
    # Price-Volume Efficiency Composite
    # Price Efficiency Component
    data['intraday_directional_bias'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['price_volume_efficiency'] = data['close'] * data['volume'] / (data['amount'] + 1e-8)
    
    # Convergence-Divergence Analysis
    # Cross-Sectional Rank Difference
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    data['rank_directional_bias'] = data.groupby(data.index)['intraday_directional_bias'].transform(cross_sectional_rank)
    data['rank_price_volume_efficiency'] = data.groupby(data.index)['price_volume_efficiency'].transform(cross_sectional_rank)
    data['cross_sectional_rank_diff'] = data['rank_directional_bias'] - data['rank_price_volume_efficiency']
    
    # Efficiency Divergence
    data['efficiency_divergence'] = data['momentum_asymmetry'] * data['cross_sectional_rank_diff']
    
    # Convergence Signal
    data['convergence_signal'] = data['volume_momentum'] * data['efficiency_divergence']
    
    # Pressure Asymmetry Composite
    # Bullish Pressure Component
    data['upward_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['volume_support'] = data['volume'] * (data['close'] > data['open'])
    data['bullish_score'] = data['upward_momentum'] * data['volume_support']
    
    # Bearish Pressure Component
    data['downward_momentum'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['volume_resistance'] = data['volume'] * (data['close'] < data['open'])
    data['bearish_score'] = data['downward_momentum'] * data['volume_resistance']
    
    # Time-Weighted Activity Adjustment
    # Recent Activity Measurement
    data['daily_activity'] = (data['high'] - data['low']) * data['volume']
    
    # Calculate exponential weights for past 5 days
    def calculate_weighted_activity(series):
        if len(series) < 5:
            return np.nan
        weights = np.exp(-np.arange(5) / 3)
        weights = weights / weights.sum()
        return np.sum(series.iloc[-5:] * weights)
    
    # Apply rolling window for recent activity
    data['recent_activity'] = data['daily_activity'].rolling(window=5, min_periods=5).apply(
        lambda x: calculate_weighted_activity(pd.Series(x)), raw=False
    )
    
    # Time-Weighted Components
    data['time_weighted_convergence'] = data['convergence_signal'] * data['recent_activity']
    data['time_weighted_pressure'] = (data['bullish_score'] - data['bearish_score']) * data['recent_activity']
    data['time_weighted_volume'] = data['volume_momentum'] * data['recent_activity']
    
    # Final Alpha Factor Generation
    # Core Factor Combination
    data['primary_factor'] = data['time_weighted_convergence'] * data['time_weighted_volume']
    data['pressure_adjusted_factor'] = data['primary_factor'] * data['time_weighted_pressure']
    data['momentum_adjusted_factor'] = data['pressure_adjusted_factor'] * data['momentum_asymmetry']
    
    # Directional Signal Enhancement
    data['volume_confirmed_factor'] = data['momentum_adjusted_factor'] * data['volume_momentum']
    
    # Final Alpha with directional bias
    data['final_alpha'] = data['volume_confirmed_factor'] * (1 + data['time_weighted_pressure'])
    
    # Return the final alpha factor series
    return data['final_alpha']
