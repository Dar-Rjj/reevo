import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required windows
    for i in range(max(15, len(data))):  # Ensure we have enough data
        if i < 14:  # Need at least 15 days for some calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # 1. Intraday Momentum Efficiency
        # Raw Momentum Strength
        high_low_ratio = (current_data['high'] - current_data['low']) / ((current_data['high'] + current_data['low']) / 2)
        momentum_3d = high_low_ratio.rolling(window=3).mean()
        
        # Trading Activity Adjustment
        volume_avg = current_data['volume'].rolling(window=5).mean()
        amount_avg = current_data['amount'].rolling(window=5).mean()
        price_efficiency = (current_data['amount'] / current_data['volume']) / ((current_data['high'] + current_data['low']) / 2)
        volume_weighted_efficiency = price_efficiency * (current_data['volume'] / volume_avg)
        
        momentum_efficiency = momentum_3d.iloc[-1] * volume_weighted_efficiency.iloc[-1]
        
        # 2. Price-Volume Fractal Coherence
        # Multi-scale Price Patterns (simplified Hurst)
        close_3d = current_data['close'].rolling(window=3).std()
        close_10d = current_data['close'].rolling(window=10).std()
        hurst_simplified = close_3d / close_10d
        
        # Volume Fractality
        volume_3d = current_data['volume'].rolling(window=3).std()
        volume_10d = current_data['volume'].rolling(window=10).std()
        volume_fractality = volume_3d / volume_10d
        
        coherence = 1 - abs(hurst_simplified.iloc[-1] - volume_fractality.iloc[-1])
        
        # 3. Overnight Gap Absorption Capacity
        gap_intensity = abs(current_data['open'] - current_data['close'].shift(1)) / current_data['close'].shift(1)
        gap_momentum = gap_intensity.rolling(window=3).mean()
        
        # Absorption Dynamics
        daily_range = current_data['high'] - current_data['low']
        gap_fill_ratio = (abs(current_data['close'] - current_data['open'])) / daily_range
        absorption_speed = 1 - gap_fill_ratio.rolling(window=3).mean()
        
        gap_absorption = gap_momentum.iloc[-1] * absorption_speed.iloc[-1]
        
        # 4. Volatility Clustering Momentum
        # Volatility Concentration
        intraday_vol = (current_data['high'] - current_data['low']) / current_data['close']
        vol_clustering = intraday_vol.rolling(window=5).std()
        
        # Clustering-Adjusted Momentum
        momentum_1d = current_data['close'].pct_change(1)
        momentum_5d = current_data['close'].pct_change(5)
        
        clustering_intensity = vol_clustering.iloc[-1] / vol_clustering.rolling(window=10).mean().iloc[-1]
        if clustering_intensity > 1:
            vol_adjusted_momentum = momentum_1d.iloc[-1]
        else:
            vol_adjusted_momentum = momentum_5d.iloc[-1]
        
        vol_cluster_momentum = vol_adjusted_momentum * clustering_intensity
        
        # 5. Price Range Expansion Probability
        # Range Expansion Patterns
        daily_range_pct = (current_data['high'] - current_data['low']) / current_data['close']
        range_expansion = daily_range_pct / daily_range_pct.rolling(window=7).mean()
        
        # Volume Expansion Confirmation
        volume_expansion = current_data['volume'] / current_data['volume'].rolling(window=15).mean()
        range_volume_corr = range_expansion.rolling(window=5).corr(volume_expansion).iloc[-1]
        
        expansion_probability = range_expansion.iloc[-1] * (1 + abs(range_volume_corr))
        
        # 6. Multi-timeframe Price Compression
        # Short-term Compression
        range_3d = (current_data['high'].rolling(window=3).max() - current_data['low'].rolling(window=3).min()) / current_data['close']
        compression_3d = 1 - (range_3d / range_3d.rolling(window=5).mean())
        
        # Medium-term Context
        range_10d = (current_data['high'].rolling(window=10).max() - current_data['low'].rolling(window=10).min()) / current_data['close']
        compression_ratio = compression_3d / (range_10d / range_10d.rolling(window=5).mean())
        
        breakout_probability = compression_ratio.iloc[-1]
        
        # 7. Volume-Weighted Price Acceleration
        # Price Momentum Acceleration
        price_change = current_data['close'].pct_change()
        price_acceleration = price_change.diff().rolling(window=5).mean()
        
        # Volume Momentum Adjustment
        volume_change = current_data['volume'].pct_change()
        volume_momentum = volume_change.rolling(window=3).mean()
        
        volume_weighted_accel = price_acceleration.iloc[-1] * (1 + volume_momentum.iloc[-1])
        
        # 8. Asymmetric Return Distribution
        # Upside/Downside Return Skew
        returns = current_data['close'].pct_change()
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            upside_skew = positive_returns.rolling(window=8).mean().iloc[-1] if not pd.isna(positive_returns.rolling(window=8).mean().iloc[-1]) else 0
            downside_skew = negative_returns.rolling(window=8).mean().iloc[-1] if not pd.isna(negative_returns.rolling(window=8).mean().iloc[-1]) else 0
            asymmetry = upside_skew - abs(downside_skew)
        else:
            asymmetry = 0
        
        # Distribution Persistence
        recent_volume = current_data['volume'].iloc[-5:].mean() / current_data['volume'].iloc[-10:-5].mean()
        asymmetric_return = asymmetry * (1 + recent_volume)
        
        # 9. Price-Level Memory Effect
        # Historical Price Attractors (simplified)
        price_levels = current_data['close'].rolling(window=20).apply(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else x.mean())
        attraction_strength = 1 / (abs(current_data['close'] - price_levels) / current_data['close'])
        
        # Current Price Proximity
        current_attraction = attraction_strength.iloc[-1] * (current_data['volume'].iloc[-1] / current_data['volume'].rolling(window=5).mean().iloc[-1])
        
        # 10. Intraday Volatility Regime Switching
        # Volatility Regime Detection
        vol_regime = intraday_vol.rolling(window=10).mean()
        current_regime = vol_regime.iloc[-1] / vol_regime.rolling(window=20).mean().iloc[-1]
        
        # Regime-Specific Signals
        if current_regime > 1.2:  # High volatility regime
            regime_signal = -momentum_1d.iloc[-1]  # Reversal focus
        elif current_regime < 0.8:  # Low volatility regime
            regime_signal = momentum_5d.iloc[-1]  # Momentum focus
        else:  # Transition regime
            regime_signal = (momentum_1d.iloc[-1] + momentum_5d.iloc[-1]) / 2  # Blend
        
        # Combine all factors with equal weighting
        combined_factor = (
            momentum_efficiency +
            coherence +
            gap_absorption +
            vol_cluster_momentum +
            expansion_probability +
            breakout_probability +
            volume_weighted_accel +
            asymmetric_return +
            current_attraction +
            regime_signal
        ) / 10
        
        factor.iloc[i] = combined_factor
    
    return factor
