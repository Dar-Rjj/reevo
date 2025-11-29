import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Volume Fractal Dynamics & Multi-factor Alpha
    Combines volatility scaling, volume distribution, momentum asymmetry, 
    auction dynamics, liquidity absorption, regime transitions, price rejection, 
    and trend exhaustion signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Volatility-Volume Fractal Dynamics
        # Multi-timeframe volatility structure
        if i >= 20:
            # Intraday volatility patterns
            high_low_range = current_data['high'] - current_data['low']
            vol_5d = high_low_range.rolling(5).std().iloc[i]
            vol_10d = high_low_range.rolling(10).std().iloc[i]
            vol_20d = high_low_range.rolling(20).std().iloc[i]
            
            # Volatility scaling properties
            vol_ratio_short_long = vol_5d / vol_20d if vol_20d > 0 else 0
            vol_persistence = vol_5d / vol_10d if vol_10d > 0 else 0
            
            # Volume distribution fractals
            if i >= 60:  # Ensure enough data for intraday patterns
                volume_data = current_data['volume']
                vol_early = volume_data.rolling(5).mean().iloc[i]  # Proxy for early session
                vol_mid = volume_data.rolling(10).mean().iloc[i]   # Proxy for mid session
                vol_late = volume_data.rolling(15).mean().iloc[i]  # Proxy for late session
                
                volume_concentration = vol_early / (vol_mid + 1e-8)
                vol_vol_similarity = (vol_early * vol_5d) / (vol_late * vol_20d + 1e-8)
            else:
                volume_concentration = 1.0
                vol_vol_similarity = 1.0
        else:
            vol_ratio_short_long = 1.0
            vol_persistence = 1.0
            volume_concentration = 1.0
            vol_vol_similarity = 1.0
        
        # Price Momentum Asymmetry Detection
        if i >= 10:
            # Directional momentum imbalance
            close_high_dist = (current_data['high'].iloc[i] - current_data['close'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 1e-8)
            close_low_dist = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 1e-8)
            
            # Momentum persistence
            up_momentum = (current_data['close'].iloc[i] > current_data['close'].iloc[i-1]) * 1.0
            down_momentum = (current_data['close'].iloc[i] < current_data['close'].iloc[i-1]) * 1.0
            
            # Volume momentum divergence
            current_volume = current_data['volume'].iloc[i]
            avg_volume = current_data['volume'].rolling(10).mean().iloc[i]
            volume_divergence = current_volume / (avg_volume + 1e-8)
            
            momentum_asymmetry = (close_high_dist - close_low_dist) * (up_momentum - down_momentum)
            volume_momentum = momentum_asymmetry * volume_divergence
        else:
            volume_momentum = 0.0
        
        # Opening Auction Imbalance Persistence
        if i >= 5:
            # Pre-open price discovery proxies
            open_price = current_data['open'].iloc[i]
            prev_close = current_data['close'].iloc[i-1]
            auction_pressure = (open_price - prev_close) / (prev_close + 1e-8)
            
            # Post-auction validation
            high_today = current_data['high'].iloc[i]
            low_today = current_data['low'].iloc[i]
            auction_level_defense = 1.0 - abs(open_price - (high_today + low_today) / 2) / ((high_today - low_today) + 1e-8)
            
            auction_strength = auction_pressure * auction_level_defense
        else:
            auction_strength = 0.0
        
        # Liquidity Absorption Dynamics
        if i >= 15:
            # Price-level liquidity measurement
            price_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            volume_today = current_data['volume'].iloc[i]
            price_movement_per_volume = price_range / (volume_today + 1e-8)
            
            # Liquidity depth analysis
            avg_volume_15d = current_data['volume'].rolling(15).mean().iloc[i]
            liquidity_depth = volume_today / (avg_volume_15d + 1e-8)
            
            # Absorption efficiency
            volume_impact = 1.0 / (price_movement_per_volume + 1e-8)
            absorption_rate = volume_impact * liquidity_depth
        else:
            absorption_rate = 0.0
        
        # Time-of-Day Regime Transition
        if i >= 30:
            # Session phase identification proxies
            morning_phase = current_data['close'].iloc[i] / current_data['open'].iloc[i] - 1.0
            afternoon_phase = current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1.0
            
            # Phase transition signals
            regime_change = abs(morning_phase - afternoon_phase)
            transition_strength = regime_change * volume_concentration
        else:
            transition_strength = 0.0
        
        # Price Rejection Strength Measurement
        if i >= 10:
            # Support/resistance testing
            recent_high = current_data['high'].rolling(10).max().iloc[i]
            recent_low = current_data['low'].rolling(10).min().iloc[i]
            current_close = current_data['close'].iloc[i]
            
            dist_to_high = (recent_high - current_close) / (recent_high - recent_low + 1e-8)
            dist_to_low = (current_close - recent_low) / (recent_high - recent_low + 1e-8)
            
            # Rejection intensity
            rejection_strength = min(dist_to_high, dist_to_low)  # Closer to extremes = stronger rejection
            
            # Volume confirmation
            rejection_volume = current_data['volume'].iloc[i] / current_data['volume'].rolling(10).mean().iloc[i]
            volume_rejection = rejection_strength * rejection_volume
        else:
            volume_rejection = 0.0
        
        # Trend Exhaustion Detection
        if i >= 20:
            # Momentum decay patterns
            returns_5d = current_data['close'].pct_change(5).iloc[i]
            returns_10d = current_data['close'].pct_change(10).iloc[i]
            momentum_decay = abs(returns_5d) / (abs(returns_10d) + 1e-8)
            
            # Volume exhaustion confirmation
            volume_trend = current_data['volume'].iloc[i] / current_data['volume'].rolling(20).mean().iloc[i]
            price_volume_divergence = abs(returns_5d) / (volume_trend + 1e-8)
            
            exhaustion_signals = momentum_decay * price_volume_divergence
        else:
            exhaustion_signals = 0.0
        
        # Composite factor combining all components
        factor_value = (
            vol_ratio_short_long * 0.15 +
            vol_persistence * 0.10 +
            volume_concentration * 0.12 +
            vol_vol_similarity * 0.08 +
            volume_momentum * 0.15 +
            auction_strength * 0.10 +
            absorption_rate * 0.10 +
            transition_strength * 0.08 +
            volume_rejection * 0.07 +
            exhaustion_signals * 0.05
        )
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
