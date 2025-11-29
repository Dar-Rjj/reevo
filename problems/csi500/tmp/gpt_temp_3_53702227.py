import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Trend Persistence Strength
    # Calculate intraday price changes
    data['intraday_high_low'] = data['high'] - data['low']
    data['open_close'] = data['close'] - data['open']
    data['abs_open_close'] = abs(data['open_close'])
    
    # Price Path Efficiency: (Close - Open) / Sum of Intraday Price Changes
    data['price_path_efficiency'] = data['open_close'] / (data['intraday_high_low'] + 1e-8)
    
    # Trend Consistency: Count of consecutive same-direction intraday bars
    data['trend_direction'] = np.sign(data['open_close'])
    data['trend_consistency'] = data['trend_direction'].groupby(level=0).transform(
        lambda x: x.expanding().apply(lambda y: (y == y.iloc[-1]).sum() if len(y) > 0 else 1)
    )
    
    # Historical trend patterns
    data['5d_avg_trend_persistence'] = data['trend_consistency'].rolling(window=5, min_periods=1).mean()
    data['persistence_ratio'] = data['trend_consistency'] / (data['5d_avg_trend_persistence'] + 1e-8)
    
    # 2. Volume-Price Divergence Signals
    # Volume-Weighted Price Change
    data['price_change'] = data['close'].pct_change()
    data['volume_weighted_price_change'] = (data['price_change'] * data['volume']).rolling(window=5, min_periods=1).sum() / (data['volume'].rolling(window=5, min_periods=1).sum() + 1e-8)
    
    # Price-Volume Correlation
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr'] = data['price_change'].rolling(window=10, min_periods=5).corr(data['volume_change'])
    
    # Divergence signals
    data['positive_divergence'] = ((data['price_change'] > 0) & (data['volume_change'] < 0)).astype(int)
    data['negative_divergence'] = ((data['price_change'] < 0) & (data['volume_change'] > 0)).astype(int)
    data['divergence_strength'] = data['positive_divergence'] - data['negative_divergence']
    
    # 3. Mean Reversion Pressure
    # Price deviation from equilibrium
    data['ma_10'] = data['close'].rolling(window=10, min_periods=5).mean()
    data['price_deviation'] = data['close'] / (data['ma_10'] + 1e-8)
    
    # Volatility-adjusted deviation
    data['std_10'] = data['close'].rolling(window=10, min_periods=5).std()
    data['volatility_adjusted_deviation'] = (data['close'] - data['ma_10']) / (data['std_10'] + 1e-8)
    
    # Reversion probability (simplified as distance from mean)
    data['reversion_probability'] = -abs(data['volatility_adjusted_deviation'])
    
    # 4. Opening Auction Dynamics (simplified for daily data)
    # For daily data, we'll use opening characteristics
    data['daily_range'] = data['high'] - data['low']
    data['open_range_ratio'] = data['abs_open_close'] / (data['daily_range'] + 1e-8)
    
    # Volume participation (using opening hour proxy)
    data['opening_volume_ratio'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean() + 1e-8)
    
    # Auction quality assessment
    data['auction_quality'] = data['open_range_ratio'] * data['opening_volume_ratio']
    
    # 5. Generate Composite Alpha Signal
    # Trend component
    trend_component = data['persistence_ratio'] * data['price_path_efficiency'] * data['price_volume_corr']
    
    # Mean reversion component
    mean_reversion_component = data['volatility_adjusted_deviation'] * data['reversion_probability']
    
    # Market microstructure component
    microstructure_component = data['auction_quality'] * data['opening_volume_ratio']
    
    # Final composite signal
    alpha_signal = (
        trend_component.fillna(0) * 0.4 +
        mean_reversion_component.fillna(0) * 0.35 +
        microstructure_component.fillna(0) * 0.25
    )
    
    return alpha_signal
