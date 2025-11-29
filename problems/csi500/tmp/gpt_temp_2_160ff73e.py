import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Frame Price-Volume Synchronization Factor
    Calculates alpha based on alignment of price momentum and volume patterns across multiple timeframes
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Temporal Frame Alignment
        # Intraday momentum
        intraday_momentum = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / \
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Overnight momentum
        overnight_momentum = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / \
                            (current_data['close'].iloc[-2] + 1e-8)
        
        # Multi-day momentum (3-day)
        multi_day_momentum = (current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / \
                            (current_data['close'].iloc[-4] + 1e-8)
        
        # Frame synchronization patterns
        frame_directions = [intraday_momentum, overnight_momentum, multi_day_momentum]
        aligned_frames = sum(1 for x in frame_directions if x > 0) - sum(1 for x in frame_directions if x < 0)
        frame_alignment_score = aligned_frames / 3.0
        
        # Volume Synchronization Analysis
        current_volume = current_data['volume'].iloc[-1]
        avg_volume_3d = current_data['volume'].iloc[-4:-1].mean()
        volume_acceleration = current_volume / (avg_volume_3d + 1e-8) - 1
        
        # Volume distribution analysis
        high_low_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        close_position = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (high_low_range + 1e-8)
        
        # Volume concentration near highs vs lows
        volume_position_score = 2 * (close_position - 0.5)  # -1 to 1 range
        
        # Price-Level Volume Anchoring
        # Previous day high/low as anchors
        prev_high = current_data['high'].iloc[-2]
        prev_low = current_data['low'].iloc[-2]
        current_close = current_data['close'].iloc[-1]
        
        # Distance to key levels
        dist_to_high = (current_close - prev_high) / (prev_high + 1e-8)
        dist_to_low = (current_close - prev_low) / (prev_low + 1e-8)
        
        # Volume at key levels
        high_volume_penetration = 1 if current_close > prev_high and volume_acceleration > 0.1 else 0
        low_volume_penetration = 1 if current_close < prev_low and volume_acceleration > 0.1 else 0
        
        # Momentum-Volume Phase Analysis
        # Volume-momentum correlation (3-day lookback)
        recent_returns = current_data['close'].iloc[-4:].pct_change().dropna()
        recent_volumes = current_data['volume'].iloc[-4:-1]
        
        if len(recent_returns) >= 2 and len(recent_volumes) >= 2:
            volume_momentum_corr = np.corrcoef(recent_returns.values, recent_volumes.values)[0,1]
            volume_momentum_corr = 0 if np.isnan(volume_momentum_corr) else volume_momentum_corr
        else:
            volume_momentum_corr = 0
        
        # Cross-Frame Liquidity Dynamics
        # Intraday liquidity concentration
        amount = current_data['amount'].iloc[-1] if 'amount' in current_data.columns else current_volume * current_data['close'].iloc[-1]
        avg_amount_3d = current_data['amount'].iloc[-4:-1].mean() if 'amount' in current_data.columns else avg_volume_3d * current_data['close'].iloc[-4:-1].mean()
        
        liquidity_concentration = amount / (avg_amount_3d + 1e-8) - 1
        
        # Synchronization Strength Quantification
        # Multi-frame alignment score
        momentum_alignment = frame_alignment_score
        
        # Volume confirmation alignment
        volume_confirmation = 1 if (volume_acceleration > 0.1 and aligned_frames > 0) or \
                                 (volume_acceleration < -0.1 and aligned_frames < 0) else 0
        
        # Liquidity pattern consistency
        liquidity_alignment = 1 if (liquidity_concentration > 0.1 and aligned_frames > 0) or \
                                 (liquidity_concentration < -0.1 and aligned_frames < 0) else 0
        
        synchronization_score = (momentum_alignment + volume_confirmation + liquidity_alignment) / 3.0
        
        # Alpha Signal Generation
        # Strong synchronization regimes
        if abs(aligned_frames) >= 2 and abs(volume_acceleration) > 0.15:
            if aligned_frames > 0:
                base_signal = 1.0  # Bullish alignment
            else:
                base_signal = -1.0  # Bearish alignment
        # Weak synchronization environments
        elif abs(aligned_frames) <= 1 and volume_momentum_corr < 0:
            # Frame conflicts with negative volume-momentum correlation
            base_signal = -aligned_frames * 0.5  # Mean reversion signal
        else:
            base_signal = aligned_frames * 0.3  # Moderate signal
        
        # Signal calibration
        volume_weight = min(abs(volume_acceleration), 1.0)
        correlation_weight = abs(volume_momentum_corr)
        synchronization_weight = abs(synchronization_score)
        
        # Final factor value
        factor_value = base_signal * (0.4 + 0.3 * volume_weight + 0.2 * correlation_weight + 0.1 * synchronization_weight)
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
