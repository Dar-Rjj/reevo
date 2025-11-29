import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor mining function
    Input: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    Output: Series of factor values indexed by date
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Based Momentum Factors
    # Overnight Gap Momentum
    data['overnight_return'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['overnight_return'])
    data['gap_persistence'] = (data['gap_direction'] == data['gap_direction'].shift(1)).astype(int)
    data['gap_persistence_count'] = data.groupby(data['gap_direction'].ne(data['gap_direction'].shift(1)).cumsum())['gap_persistence'].cumsum()
    data['overnight_gap_momentum'] = data['overnight_return'] * (1 + data['gap_persistence_count'])
    
    # Intraday Range Efficiency
    data['daily_range'] = data['high'] - data['low']
    data['absolute_return'] = abs(data['close'] - data['open'])
    data['range_efficiency'] = data['absolute_return'] / data['daily_range'].replace(0, np.nan)
    
    # Price Compression Signal
    data['range_percentile'] = data['daily_range'].rolling(window=20, min_periods=10).apply(lambda x: np.percentile(x.dropna(), 30), raw=False)
    data['narrow_range'] = (data['daily_range'] < data['range_percentile']).astype(int)
    data['compression_breakout'] = data['narrow_range'].shift(1) * (data['close'] / data['open'] - 1)
    data['compression_expansion'] = data['compression_breakout'].rolling(window=5, min_periods=3).mean()
    
    # Volume-Price Interaction Factors
    # Volume-Adjusted Range Momentum
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['volume_adjusted_momentum'] = data['daily_range'] * data['volume_ratio']
    
    # Liquidity-Efficient Movement
    data['price_change'] = data['close'] - data['open']
    data['price_move_efficiency'] = abs(data['price_change']) / data['volume'].replace(0, np.nan)
    data['volume_quantile'] = data['volume'].rolling(window=20, min_periods=10).apply(lambda x: np.percentile(x.dropna(), 30), raw=False)
    data['low_volume_high_efficiency'] = (data['volume'] < data['volume_quantile']) * data['price_move_efficiency']
    
    # Volume-Price Divergence
    data['price_momentum'] = data['close'].pct_change(periods=3)
    data['volume_momentum'] = data['volume'].pct_change(periods=3)
    data['volume_price_divergence'] = data['price_momentum'] - data['volume_momentum']
    
    # Volatility Regime Factors
    # Volatility Transition Detector
    data['range_volatility'] = data['daily_range'].rolling(window=20, min_periods=10).std()
    data['volatility_regime'] = (data['range_volatility'] > data['range_volatility'].rolling(window=40, min_periods=20).mean()).astype(int)
    data['volatility_breakout'] = data['volatility_regime'].diff() * data['price_change']
    
    # Intraday Volatility Pattern
    data['morning_volatility'] = (data['high'] - data['open']).abs() + (data['low'] - data['open']).abs()
    data['afternoon_volatility'] = (data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min())
    data['volatility_persistence'] = data['morning_volatility'].rolling(window=5, min_periods=3).corr(data['afternoon_volatility'])
    
    # Market Microstructure Factors
    # Opening Auction Strength (using first available data as proxy)
    data['opening_range'] = data['high'] - data['low']
    data['opening_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    data['opening_strength'] = data['opening_range'] * data['opening_volume_ratio']
    
    # Session Transition Momentum
    data['morning_trend'] = data['close'] - data['open']
    data['afternoon_trend'] = data['close'] - data['close'].shift(1)
    data['trend_consistency'] = np.sign(data['morning_trend']) == np.sign(data['afternoon_trend'])
    data['overnight_momentum'] = data['open'] - data['close'].shift(1)
    data['intraday_momentum'] = data['close'] - data['open']
    data['momentum_transfer'] = np.sign(data['overnight_momentum']) == np.sign(data['intraday_momentum'])
    data['multi_session_persistence'] = (data['trend_consistency'].astype(int) + data['momentum_transfer'].astype(int)) / 2
    
    # Combine all factors with appropriate weights
    factors = [
        data['overnight_gap_momentum'],
        data['range_efficiency'],
        data['compression_expansion'],
        data['volume_adjusted_momentum'],
        data['low_volume_high_efficiency'],
        data['volume_price_divergence'],
        data['volatility_breakout'],
        data['volatility_persistence'],
        data['opening_strength'],
        data['multi_session_persistence']
    ]
    
    # Normalize each factor and combine
    normalized_factors = []
    for factor in factors:
        normalized = (factor - factor.rolling(window=60, min_periods=20).mean()) / factor.rolling(window=60, min_periods=20).std()
        normalized_factors.append(normalized)
    
    # Equal weighted combination
    combined_factor = pd.concat(normalized_factors, axis=1).mean(axis=1)
    
    return combined_factor
