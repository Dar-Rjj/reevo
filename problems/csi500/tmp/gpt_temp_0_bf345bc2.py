import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration with Volatility Adjustment
    # Calculate Intraday Momentum Components
    morning_momentum = (data['high'] - data['open']) / data['open']
    afternoon_momentum = (data['close'] - data['low']) / data['low']
    
    # Compute Volatility Normalization
    intraday_range = data['high'] - data['low']
    
    # Calculate ATR components
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    # True range calculation
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=5).mean()
    
    # Volatility Adjustment Factor
    volatility_adjustment = intraday_range / atr
    volatility_adjustment = volatility_adjustment.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Generate Volatility-Adjusted Momentum
    vol_adj_morning = morning_momentum / volatility_adjustment
    vol_adj_afternoon = afternoon_momentum / volatility_adjustment
    momentum_acceleration = vol_adj_afternoon - vol_adj_morning
    
    # Trend Persistence with Efficiency Filters
    # Multi-timeframe Trend Analysis
    intraday_trend = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_trend = intraday_trend.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    short_term_trend = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    trend_convergence = np.sign(intraday_trend) * np.sign(short_term_trend)
    
    # Price Efficiency Measures
    efficiency_ratio = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    momentum_consistency = np.sign(morning_momentum) * np.sign(afternoon_momentum)
    
    # Trend Strength Assessment
    momentum_magnitude = abs(vol_adj_morning) + abs(vol_adj_afternoon)
    
    breakout_confirmation = ((data['close'] > data['high'].shift(1)) | 
                           (data['close'] < data['low'].shift(1))).astype(int)
    
    # Trend Persistence Score
    daily_returns = data['close'].pct_change()
    trend_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            recent_returns = daily_returns.iloc[i-2:i+1]  # Current day + 2 previous days
            same_sign_count = (np.sign(recent_returns) == np.sign(daily_returns.iloc[i])).sum()
            trend_persistence.iloc[i] = same_sign_count
        else:
            trend_persistence.iloc[i] = 1
    
    # Liquidity-Price Divergence Framework
    # VWAP-Based Price Analysis
    vwap = data['amount'] / data['volume']
    vwap = vwap.replace([np.inf, -np.inf], np.nan).fillna(data['close'])
    
    price_vwap_divergence = data['close'] - vwap
    normalized_divergence = price_vwap_divergence / data['close']
    
    # Volume Dynamics Assessment
    volume_velocity = data['volume'] / data['volume'].shift(1)
    volume_velocity = volume_velocity.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    volume_change_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_change_ratio = volume_change_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    amount_intensity = data['amount'] / data['amount'].rolling(window=5).median()
    amount_intensity = amount_intensity.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Liquidity Confirmation Signals
    liquidity_momentum = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    liquidity_momentum = liquidity_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # High-Low Liquidity Distribution (simplified)
    mid_price = (data['high'] + data['low']) / 2
    high_price_liquidity = data['volume'].where(data['close'] > mid_price, 0)
    low_price_liquidity = data['volume'].where(data['close'] < mid_price, 0)
    
    liquidity_divergence_signal = price_vwap_divergence * volume_change_ratio
    
    # Market Regime Adaptive Weighting
    # Market Condition Classification
    trend_strength = abs(short_term_trend)
    
    volatility_regime = ((data['high'] - data['low']) / data['close']).rolling(window=10).std()
    volatility_regime_median = volatility_regime.rolling(window=20).median()
    
    liquidity_regime = volume_velocity.rolling(window=5).std()
    liquidity_regime_median = liquidity_regime.rolling(window=20).median()
    
    # Regime-Based Signal Enhancement
    trending_market = (trend_strength > 0.02) & (momentum_consistency > 0)
    volatile_market = volatility_regime > (1.5 * volatility_regime_median)
    liquid_market = liquidity_regime < (0.7 * liquidity_regime_median)
    
    # Regime multipliers
    regime_multiplier = pd.Series(1.0, index=data.index)
    regime_multiplier[trending_market] = 1.4
    regime_multiplier[volatile_market] = 0.7
    regime_multiplier[liquid_market] = 1.2
    
    # Composite Factor Generation
    # Core Momentum-Volatility Component
    base_signal = momentum_acceleration * trend_convergence * efficiency_ratio
    volatility_adjusted_signal = base_signal * (1 / volatility_adjustment)
    
    # Liquidity Confirmation Layer
    liquidity_weight = amount_intensity * np.sign(liquidity_divergence_signal)
    confirmed_signal = volatility_adjusted_signal * liquidity_weight
    
    # Apply Breakout Confirmation filter for extreme signals
    breakout_multiplier = pd.Series(1.0, index=data.index)
    extreme_signals = abs(confirmed_signal) > confirmed_signal.rolling(window=20).quantile(0.9)
    breakout_multiplier[extreme_signals & (breakout_confirmation == 0)] = 0.5
    
    # Final Factor Construction
    final_factor = (confirmed_signal * regime_multiplier * 
                   trend_persistence * breakout_multiplier)
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return final_factor
