import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate multiple alpha factors based on price, volume, and amount data.
    Returns a composite factor series.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Range-Volume Divergence
    # Calculate daily range percentage
    data['daily_range_pct'] = (data['high'] - data['low']) / data['low']
    
    # Volume change from previous day
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume spikes (z-score based)
    volume_rolling_mean = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_rolling_std = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_zscore'] = (data['volume'] - volume_rolling_mean) / volume_rolling_std
    
    # Range-Volume Divergence
    range_rank = data['daily_range_pct'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_rank = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['range_volume_divergence'] = range_rank - volume_rank
    
    # Factor 2: Amount-Weighted Gap Momentum
    # Opening gap percentage
    data['gap_pct'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Amount-to-volume ratio
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    
    # Amount momentum (3-day)
    data['amount_momentum'] = data['amount'] / data['amount'].rolling(window=3, min_periods=2).mean() - 1
    
    # Gap momentum with amount confirmation
    gap_direction = np.sign(data['gap_pct'])
    amount_confirmation = data['amount_momentum'] * gap_direction
    data['gap_amount_momentum'] = data['gap_pct'] * (1 + amount_confirmation)
    
    # Factor 3: Volatility-Adjusted Price-Volume Efficiency
    # Price efficiency ratio (close-to-close vs high-to-low)
    price_efficiency = (data['close'] - data['close'].shift(1)).abs() / (data['high'] - data['low'])
    
    # Volatility context (current range vs 20-day average range)
    avg_range = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    volatility_ratio = (data['high'] - data['low']) / avg_range
    
    # Volume efficiency (volume per price movement)
    price_movement = (data['high'] - data['low']).replace(0, np.nan)
    volume_efficiency = data['volume'] / price_movement
    
    # Combined efficiency factor
    data['vol_adj_efficiency'] = price_efficiency * volatility_ratio / (volume_efficiency.rolling(window=10, min_periods=5).mean())
    
    # Factor 4: Multi-Timeframe Pressure Divergence
    # Intraday buying pressure (close relative to range)
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 3-day momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Volume intensity (current volume vs 5-day average)
    volume_intensity = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Pressure divergence
    pressure_rank = data['buying_pressure'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    momentum_rank = data['momentum_3d'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['pressure_divergence'] = (pressure_rank - momentum_rank) * volume_intensity
    
    # Factor 5: High-Low Breakout with Amount Confirmation
    # 5-day average range
    avg_5d_range = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    
    # Range expansion (current range vs average)
    range_expansion = (data['high'] - data['low']) / avg_5d_range - 1
    
    # Amount momentum for breakout validation
    amount_momentum_5d = data['amount'] / data['amount'].rolling(window=5, min_periods=3).mean() - 1
    
    # Breakout factor
    data['breakout_amount'] = range_expansion * amount_momentum_5d
    
    # Factor 6: Price Elasticity with Volume Response
    # Small oscillation patterns (2-day price changes)
    price_oscillation = (data['close'] - data['close'].shift(2)).abs() / data['close'].shift(2)
    
    # Volume response (current volume vs 2-day average)
    volume_response = data['volume'] / data['volume'].rolling(window=2, min_periods=2).mean()
    
    # Price elasticity (volume response to price changes)
    data['price_elasticity'] = volume_response / (price_oscillation.replace(0, np.nan) + 0.001)
    
    # Factor 7: Liquidity-Regime Switching
    # Liquidity metrics
    avg_amount = data['amount'].rolling(window=20, min_periods=10).mean()
    liquidity_ratio = data['amount'] / avg_amount
    
    # Identify liquidity regimes (z-score based)
    amount_zscore = (data['amount'] - data['amount'].rolling(window=20, min_periods=10).mean()
                    ) / data['amount'].rolling(window=20, min_periods=10).std()
    
    # Price behavior in different regimes
    high_liquidity_signal = (amount_zscore > 1) & (data['daily_range_pct'] < data['daily_range_pct'].rolling(window=20).quantile(0.3))
    low_liquidity_signal = (amount_zscore < -0.5) & (data['daily_range_pct'] > data['daily_range_pct'].rolling(window=20).quantile(0.7))
    
    data['liquidity_regime'] = 0
    data.loc[high_liquidity_signal, 'liquidity_regime'] = -1  # Lower alpha in high liquidity
    data.loc[low_liquidity_signal, 'liquidity_regime'] = 1   # Higher alpha in low liquidity
    
    # Factor 8: Opening Range Breakout
    # Since we don't have intraday data, approximate with daily open and close
    opening_strength = (data['close'] - data['open']) / data['open']
    
    # Volume confirmation (opening volume intensity)
    # Use first available volume as proxy for opening volume
    volume_intensity_open = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    data['opening_breakout'] = opening_strength * volume_intensity_open
    
    # Combine all factors with equal weights
    factors = [
        'range_volume_divergence',
        'gap_amount_momentum', 
        'vol_adj_efficiency',
        'pressure_divergence',
        'breakout_amount',
        'price_elasticity',
        'liquidity_regime',
        'opening_breakout'
    ]
    
    # Normalize each factor by its rolling z-score
    composite_factor = pd.Series(0, index=data.index)
    valid_factors = 0
    
    for factor in factors:
        if factor in data.columns:
            # Remove outliers and normalize
            factor_data = data[factor].copy()
            rolling_mean = factor_data.rolling(window=20, min_periods=10).mean()
            rolling_std = factor_data.rolling(window=20, min_periods=10).std()
            normalized_factor = (factor_data - rolling_mean) / rolling_std
            
            # Winsorize extreme values
            normalized_factor = normalized_factor.clip(lower=-3, upper=3)
            
            composite_factor += normalized_factor
            valid_factors += 1
    
    if valid_factors > 0:
        composite_factor = composite_factor / valid_factors
    
    return composite_factor
