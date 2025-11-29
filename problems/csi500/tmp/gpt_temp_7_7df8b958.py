import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Momentum with Volume Confirmation
    # Calculate High-Low Range Momentum
    high_low_range = data['high'] - data['low']
    high_low_momentum = (high_low_range / high_low_range.shift(1) - 1).fillna(0)
    
    # Confirm with Volume Pattern
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_trend = (volume_ma_5 / volume_ma_20 - 1).fillna(0)
    volume_direction = np.sign(data['volume'].diff().fillna(0))
    volume_consistency = volume_direction.rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    ).fillna(0)
    
    hl_momentum_factor = high_low_momentum * (1 + volume_trend) * volume_consistency
    
    # Price Reversal After Extreme Volume
    # Detect Extreme Volume Events
    volume_ma_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_std_10 = data['volume'].rolling(window=10, min_periods=1).std()
    volume_zscore = (data['volume'] - volume_ma_10) / (volume_std_10 + 1e-8)
    volume_spike = (volume_zscore > 2).astype(int)
    
    # Measure Subsequent Price Action
    returns_1d = data['close'].pct_change(1).fillna(0)
    returns_3d = data['close'].pct_change(3).fillna(0)
    
    # Create lagged volume spike indicator
    volume_spike_lagged = volume_spike.shift(1).fillna(0)
    reversal_factor = -volume_spike_lagged * returns_1d + (1 - volume_spike_lagged) * returns_3d
    
    # Opening Gap Persistence Factor
    # Calculate Opening Gap
    gap_percentage = (data['open'] / data['close'].shift(1) - 1).fillna(0)
    
    # Measure Gap Persistence
    intraday_high = data['high']
    intraday_low = data['low']
    gap_filled = ((gap_percentage > 0) & (intraday_low <= data['close'].shift(1))) | \
                 ((gap_percentage < 0) & (intraday_high >= data['close'].shift(1)))
    gap_fill_indicator = (~gap_filled).astype(int)
    
    # Volume Support Assessment
    volume_rank = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 60)) if len(x) > 0 else 0, raw=False
    ).fillna(0)
    
    gap_persistence_factor = gap_percentage * gap_fill_indicator * volume_rank
    
    # Amount-Based Price Efficiency
    # Analyze Amount Patterns
    amount_change = data['amount'].pct_change().fillna(0)
    amount_trend = data['amount'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    # Relate to Price Movements
    price_change = data['close'].pct_change().fillna(0)
    price_efficiency = price_change / (np.abs(amount_change) + 1e-8)
    efficiency_consistency = price_efficiency.rolling(window=5, min_periods=1).std().fillna(0)
    
    amount_efficiency_factor = price_efficiency / (efficiency_consistency + 1e-8) * np.sign(amount_trend)
    
    # Multi-Timeframe Volatility Convergence
    # Calculate Short-term Volatility (using High-Low range)
    short_term_vol = high_low_range.rolling(window=5, min_periods=1).std().fillna(0)
    
    # Calculate Medium-term Volatility (using Close returns)
    medium_term_vol = data['close'].pct_change().rolling(window=20, min_periods=1).std().fillna(0)
    
    # Volatility Ratio Analysis
    vol_ratio = short_term_vol / (medium_term_vol + 1e-8)
    vol_convergence = (vol_ratio - vol_ratio.rolling(window=10, min_periods=1).mean()).fillna(0)
    
    volatility_factor = -vol_convergence  # Negative because convergence might mean decreasing volatility
    
    # Volume-Weighted Price Range Efficiency
    # Compute Price Range Efficiency
    close_open_range = data['close'] - data['open']
    range_utilization = close_open_range / (high_low_range + 1e-8)
    
    # Apply Volume Weighting
    volume_rank_vw = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - np.mean(x)) / (np.std(x) + 1e-8) if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    volume_weight = 1 + np.tanh(volume_rank_vw / 2)  # Scale between 0.5 and 1.5
    range_efficiency_factor = range_utilization * volume_weight
    
    # Intraday Momentum Carryover
    # Measure Intraday Momentum
    intraday_return = (data['close'] - data['open']) / data['open']
    intraday_strength = intraday_return.rolling(window=5, min_periods=1).mean().fillna(0)
    
    # Assess Next Day Impact (using same-day momentum for prediction)
    momentum_persistence = intraday_return * intraday_strength
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'hl_momentum': hl_momentum_factor,
        'reversal': reversal_factor,
        'gap_persistence': gap_persistence_factor,
        'amount_efficiency': amount_efficiency_factor,
        'volatility': volatility_factor,
        'range_efficiency': range_efficiency_factor,
        'momentum_persistence': momentum_persistence
    })
    
    # Z-score normalization for each factor
    for col in factors.columns:
        factors[col] = (factors[col] - factors[col].rolling(window=20, min_periods=1).mean()) / \
                      (factors[col].rolling(window=20, min_periods=1).std() + 1e-8)
    
    # Equal-weighted combination
    combined_factor = factors.mean(axis=1)
    
    return combined_factor
