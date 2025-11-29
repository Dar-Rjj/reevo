import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Range Breakout Momentum
    # Calculate Daily High-Low Range
    data['high_low_range'] = data['high'] - data['low']
    data['prev_high_low_range'] = data['high_low_range'].shift(1)
    
    # Detect Range Breakout
    data['breakout_ratio'] = np.where(
        data['high'] > data['high'].shift(1) + data['prev_high_low_range'] * 0.5,
        (data['high'] - (data['high'].shift(1) + data['prev_high_low_range'] * 0.5)) / data['prev_high_low_range'],
        0
    )
    breakout_factor = data['breakout_ratio'] * (data['volume'] / data['volume'].rolling(5).mean())
    
    # Volume-Adjusted Price Acceleration
    # Calculate Price Momentum
    data['three_day_returns'] = data['close'].pct_change(3)
    data['momentum_acceleration'] = data['three_day_returns'] - data['three_day_returns'].shift(1)
    
    # Adjust by Volume Pattern
    data['volume_trend'] = data['volume'] / data['volume'].rolling(5).mean()
    price_accel_factor = data['momentum_acceleration'] * data['volume_trend']
    
    # Open-Gap Reversal Probability
    # Calculate Opening Gap
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Assess Gap Filling Behavior
    data['intraday_movement'] = (data['close'] - data['open']) / data['open']
    data['gap_persistence'] = np.where(
        data['opening_gap'] * data['intraday_movement'] < 0,
        abs(data['opening_gap']) - abs(data['intraday_movement']),
        0
    )
    reversal_factor = -data['gap_persistence'] * (data['volume'] / data['volume'].rolling(5).mean())
    
    # Amount-Based Price Efficiency
    # Analyze Trading Amount Patterns
    data['amount_volatility'] = data['amount'].rolling(5).std() / data['amount'].rolling(5).mean()
    
    # Relate to Price Movement Efficiency
    data['price_change_per_amount'] = (data['close'] - data['close'].shift(1)).abs() / data['amount']
    efficiency_factor = -data['price_change_per_amount'] * data['amount_volatility']
    
    # Multi-Timeframe Volume-Price Divergence
    # Short-Term Analysis (1-3 days)
    data['short_volume_trend'] = data['volume'] / data['volume'].rolling(3).mean()
    data['short_price_momentum'] = data['close'].pct_change(3)
    
    # Medium-Term Analysis (5-10 days)
    data['medium_volume_trend'] = data['volume'] / data['volume'].rolling(10).mean()
    data['medium_price_momentum'] = data['close'].pct_change(10)
    
    # Detect Divergence Patterns
    volume_divergence = data['short_volume_trend'] - data['medium_volume_trend']
    price_divergence = data['short_price_momentum'] - data['medium_price_momentum']
    divergence_factor = volume_divergence * price_divergence
    
    # Intraday Volatility Persistence
    # Calculate True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Analyze Volatility Clustering
    data['volatility_regime'] = data['true_range'] / data['true_range'].rolling(10).mean()
    data['volatility_autocorr'] = data['true_range'].rolling(5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    volatility_factor = data['volatility_autocorr'] * data['volatility_regime'] * (data['volume'] / data['volume'].rolling(5).mean())
    
    # Price-Volume Fractal Efficiency
    # Calculate Fractal Dimension using High-Low-Close patterns
    data['price_range'] = data['high'] - data['low']
    data['price_complexity'] = (abs(data['close'] - data['open']) + data['price_range']) / data['price_range']
    
    # Combine with Volume Distribution
    data['volume_fractal'] = data['volume'].rolling(5).apply(
        lambda x: np.log(len(x)) / np.log(np.std(x) + 1) if np.std(x) > 0 else 1
    )
    
    # Generate Efficiency Score
    efficiency_score = data['price_complexity'] / (data['volume_fractal'] + 1e-6)
    fractal_factor = -efficiency_score.rolling(5).mean()
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'breakout': breakout_factor,
        'price_accel': price_accel_factor,
        'reversal': reversal_factor,
        'efficiency': efficiency_factor,
        'divergence': divergence_factor,
        'volatility': volatility_factor,
        'fractal': fractal_factor
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
