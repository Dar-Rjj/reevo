import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate multiple alpha factors using only current and historical data.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
        Index should be datetime
    
    Returns:
    Series: Combined alpha factor values indexed by date
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor storage
    factors = pd.Series(index=data.index, dtype=float)
    
    # Factor 1: High-Low Range Breakout Momentum
    # Calculate daily high-low range
    daily_range = data['high'] - data['low']
    
    # Previous day's range maximum (high of previous day)
    prev_range_max = data['high'].shift(1)
    
    # Breakout ratio: current high relative to previous range maximum
    breakout_ratio = (data['high'] / prev_range_max) - 1.0
    
    # Scale by volume confirmation
    volume_normalized = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    factor1 = breakout_ratio * volume_normalized
    
    # Factor 2: Volume-Adjusted Price Acceleration
    # Three-day returns
    returns_3d = data['close'].pct_change(periods=3)
    
    # Momentum acceleration (second derivative of returns)
    momentum_accel = returns_3d.diff()
    
    # Volume trend analysis
    volume_ma_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    volume_trend = volume_ma_ratio.rolling(window=3, min_periods=1).mean()
    
    # Combine acceleration with volume trend
    factor2 = momentum_accel * volume_trend
    
    # Factor 3: Open-Gap Reversal Probability
    # Calculate opening gap
    opening_gap = (data['open'] / data['close'].shift(1)) - 1.0
    
    # Gap size vs intraday range
    intraday_range = (data['high'] - data['low']) / data['open']
    gap_persistence = abs(opening_gap) / (intraday_range + 1e-8)
    
    # Volume during gap period (current day volume)
    gap_volume = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Reversal signal (negative relationship: larger gaps with high volume more likely to reverse)
    factor3 = -opening_gap * gap_persistence * gap_volume
    
    # Factor 4: Amount-Based Price Efficiency
    # Amount volatility (rolling standard deviation normalized by average)
    amount_vol = data['amount'].rolling(window=10, min_periods=1).std()
    amount_avg = data['amount'].rolling(window=10, min_periods=1).mean()
    amount_vol_normalized = amount_vol / (amount_avg + 1e-8)
    
    # Price change per unit amount
    price_change = data['close'].diff().abs()
    price_efficiency = price_change / (data['amount'] + 1e-8)
    
    # Inefficient pricing signal (high amount volatility with low price efficiency)
    factor4 = -amount_vol_normalized * price_efficiency.rolling(window=5, min_periods=1).mean()
    
    # Factor 5: Multi-Timeframe Volume-Price Divergence
    # Short-term analysis (3 days)
    short_volume_trend = data['volume'].pct_change(periods=3)
    short_price_momentum = data['close'].pct_change(periods=3)
    
    # Medium-term analysis (8 days)
    medium_volume_trend = data['volume'].pct_change(periods=8)
    medium_price_momentum = data['close'].pct_change(periods=8)
    
    # Divergence patterns
    volume_divergence = short_volume_trend - medium_volume_trend
    price_divergence = short_price_momentum - medium_price_momentum
    
    # Convergence signals (negative when diverging)
    factor5 = -(volume_divergence * price_divergence)
    
    # Factor 6: Intraday Volatility Persistence
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volatility autocorrelation (1-day lag)
    vol_autocorr = true_range.rolling(window=10, min_periods=1).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Volume confirmation during volatility periods
    vol_period_volume = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Persistence signal
    factor6 = vol_autocorr * vol_period_volume
    
    # Factor 7: Price-Volume Fractal Efficiency
    # Price path complexity (simplified fractal measure)
    price_high_low_range = (data['high'] - data['low']) / data['close']
    price_close_range = abs(data['close'] - data['open']) / data['close']
    price_complexity = (price_high_low_range + price_close_range).rolling(window=5, min_periods=1).std()
    
    # Volume pattern complexity
    volume_complexity = data['volume'].pct_change().abs().rolling(window=5, min_periods=1).std()
    
    # Efficiency score (difference between price and volume complexity patterns)
    factor7 = price_complexity - volume_complexity
    
    # Combine all factors with equal weights
    all_factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5,
        'factor6': factor6,
        'factor7': factor7
    })
    
    # Z-score normalization for each factor
    for col in all_factors.columns:
        all_factors[col] = (all_factors[col] - all_factors[col].rolling(window=20, min_periods=1).mean()) / (all_factors[col].rolling(window=20, min_periods=1).std() + 1e-8)
    
    # Equal-weighted combination
    combined_factor = all_factors.mean(axis=1)
    
    return combined_factor
