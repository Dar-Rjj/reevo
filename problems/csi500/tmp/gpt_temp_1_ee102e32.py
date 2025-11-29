import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Alpha Factor combining behavioral momentum patterns from gap absorption,
    intraday pressure accumulation, compression release, efficiency gap convergence,
    and range expansion dynamics.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Required minimum data points
    min_periods = 20
    
    for i in range(min_periods, len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Gap Absorption Momentum with Volume Confirmation
        gap_absorption_signal = _calculate_gap_absorption(current_data)
        
        # 2. Intraday Pressure Accumulation with Range Efficiency
        pressure_signal = _calculate_pressure_accumulation(current_data)
        
        # 3. Compression Release Momentum with Volume Acceleration
        compression_signal = _calculate_compression_release(current_data)
        
        # 4. Efficiency Gap Convergence with Pressure Divergence
        efficiency_signal = _calculate_efficiency_convergence(current_data)
        
        # 5. Range Expansion Momentum with Efficiency Confirmation
        expansion_signal = _calculate_range_expansion(current_data)
        
        # Combine all signals with equal weighting
        combined_signal = (
            gap_absorption_signal + 
            pressure_signal + 
            compression_signal + 
            efficiency_signal + 
            expansion_signal
        ) / 5.0
        
        factor.iloc[i] = combined_signal
    
    # Handle initial NaN values
    factor = factor.fillna(0)
    
    return factor

def _calculate_gap_absorption(data):
    """Calculate Gap Absorption Momentum with Volume Confirmation"""
    if len(data) < 3:
        return 0
    
    current = data.iloc[-1]
    prev_close = data.iloc[-2]['close']
    
    # Opening Gap
    opening_gap = (current['open'] - prev_close) / prev_close
    
    # Gap Recovery
    gap_magnitude = abs(current['open'] - prev_close)
    if gap_magnitude > 0:
        gap_recovery = (current['close'] - current['open']) / gap_magnitude
    else:
        gap_recovery = 0
    
    # Absorption Strength (positive when gap is filled)
    absorption_strength = gap_recovery * np.sign(opening_gap)
    
    # Volume Confirmation
    if len(data) >= 20:
        median_volume = data['volume'].iloc[-20:].median()
        volume_intensity = current['volume'] / median_volume if median_volume > 0 else 1
    else:
        volume_intensity = 1
    
    # Persistence over 2-4 days
    if len(data) >= 5:
        recent_absorption = []
        for j in range(2, min(5, len(data))):
            if len(data) > j:
                prev_data = data.iloc[-(j+1)]
                prev_prev_close = data.iloc[-(j+2)]['close'] if len(data) > j+1 else prev_data['open']
                gap_mag = abs(prev_data['open'] - prev_prev_close)
                if gap_mag > 0:
                    rec = (prev_data['close'] - prev_data['open']) / gap_mag
                    recent_absorption.append(rec * np.sign((prev_data['open'] - prev_prev_close) / prev_prev_close))
        
        persistence = np.mean(recent_absorption) if recent_absorption else 0
    else:
        persistence = 0
    
    signal = absorption_strength * volume_intensity * (1 + 0.5 * persistence)
    return signal

def _calculate_pressure_accumulation(data):
    """Calculate Intraday Pressure Accumulation with Range Efficiency"""
    if len(data) < 3:
        return 0
    
    current = data.iloc[-1]
    
    # Buying Pressure
    price_range = current['high'] - current['low']
    if price_range > 0:
        buying_pressure = (current['close'] - current['low']) / price_range
    else:
        buying_pressure = 0.5
    
    # Pressure Persistence (3-day sum)
    pressure_sum = buying_pressure
    for j in range(1, min(3, len(data))):
        prev_data = data.iloc[-(j+1)]
        prev_range = prev_data['high'] - prev_data['low']
        if prev_range > 0:
            prev_pressure = (prev_data['close'] - prev_data['low']) / prev_range
            pressure_sum += prev_pressure
    
    # Range Efficiency
    if price_range > 0:
        range_utilization = abs(current['close'] - current['open']) / price_range
    else:
        range_utilization = 0
    
    # Efficiency Momentum
    if len(data) >= 6:
        recent_efficiency = []
        for j in range(5):
            if len(data) > j+1:
                prev_data = data.iloc[-(j+2)]
                prev_range = prev_data['high'] - prev_data['low']
                if prev_range > 0:
                    eff = abs(prev_data['close'] - prev_data['open']) / prev_range
                    recent_efficiency.append(eff)
        
        avg_efficiency = np.mean(recent_efficiency) if recent_efficiency else range_utilization
        efficiency_momentum = range_utilization - avg_efficiency
    else:
        efficiency_momentum = 0
    
    signal = pressure_sum * (1 + efficiency_momentum)
    return signal

def _calculate_compression_release(data):
    """Calculate Compression Release Momentum with Volume Acceleration"""
    if len(data) < 9:
        return 0
    
    current = data.iloc[-1]
    
    # Compression Ratio
    avg_price = (current['high'] + current['low']) / 2
    if avg_price > 0:
        compression_ratio = (current['high'] - current['low']) / avg_price
    else:
        compression_ratio = 0
    
    # Detect compression (below 8-day median)
    if len(data) >= 9:
        recent_ranges = []
        for j in range(8):
            prev_data = data.iloc[-(j+2)]
            prev_avg = (prev_data['high'] + prev_data['low']) / 2
            if prev_avg > 0:
                ratio = (prev_data['high'] - prev_data['low']) / prev_avg
                recent_ranges.append(ratio)
        
        median_compression = np.median(recent_ranges) if recent_ranges else compression_ratio
        is_compressed = compression_ratio < median_compression * 0.8
    else:
        is_compressed = False
    
    # Release detection and strength
    if is_compressed and len(data) >= 2:
        prev_data = data.iloc[-2]
        prev_avg = (prev_data['high'] + prev_data['low']) / 2
        if prev_avg > 0:
            prev_compression = (prev_data['high'] - prev_data['low']) / prev_avg
            release_strength = (compression_ratio - prev_compression) / prev_compression if prev_compression > 0 else 0
        else:
            release_strength = 0
    else:
        release_strength = 0
    
    # Volume Acceleration
    if len(data) >= 4:
        recent_volumes = []
        for j in range(3):
            if len(data) > j+1:
                recent_volumes.append(data.iloc[-(j+2)]['volume'])
        
        avg_volume = np.mean(recent_volumes) if recent_volumes else current['volume']
        volume_momentum = current['volume'] / avg_volume if avg_volume > 0 else 1
    else:
        volume_momentum = 1
    
    # Price momentum for directional confirmation
    if len(data) >= 2:
        price_momentum = (current['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']
    else:
        price_momentum = 0
    
    signal = release_strength * volume_momentum * (1 + price_momentum)
    return signal

def _calculate_efficiency_convergence(data):
    """Calculate Efficiency Gap Convergence with Pressure Divergence"""
    if len(data) < 6:
        return 0
    
    current = data.iloc[-1]
    
    # Flow Efficiency
    if current['close'] > 0 and current['volume'] > 0:
        flow_efficiency = current['amount'] / (current['close'] * current['volume'])
    else:
        flow_efficiency = 0
    
    # Efficiency Gap
    if len(data) >= 6:
        recent_efficiency = []
        for j in range(5):
            if len(data) > j+1:
                prev_data = data.iloc[-(j+2)]
                if prev_data['close'] > 0 and prev_data['volume'] > 0:
                    eff = prev_data['amount'] / (prev_data['close'] * prev_data['volume'])
                    recent_efficiency.append(eff)
        
        avg_efficiency = np.mean(recent_efficiency) if recent_efficiency else flow_efficiency
        efficiency_gap = flow_efficiency - avg_efficiency
    else:
        efficiency_gap = 0
    
    # Net Pressure
    price_range = current['high'] - current['low']
    if price_range > 0:
        net_pressure = (2 * current['close'] - current['high'] - current['low']) / price_range
    else:
        net_pressure = 0
    
    # Pressure Divergence
    pressure_divergence = net_pressure * np.sign(efficiency_gap)
    
    # Volume trend confirmation
    if len(data) >= 6:
        recent_volumes = [data.iloc[-(j+1)]['volume'] for j in range(5)]
        volume_trend = np.polyfit(range(5), recent_volumes, 1)[0] / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 0
    else:
        volume_trend = 0
    
    # Range scaling
    avg_price = (current['high'] + current['low']) / 2
    if avg_price > 0:
        daily_range = (current['high'] - current['low']) / avg_price
        range_scaling = 1 + daily_range
    else:
        range_scaling = 1
    
    signal = efficiency_gap * pressure_divergence * range_scaling * (1 + 0.2 * volume_trend)
    return signal

def _calculate_range_expansion(data):
    """Calculate Range Expansion Momentum with Efficiency Confirmation"""
    if len(data) < 6:
        return 0
    
    current = data.iloc[-1]
    
    # Daily Range
    avg_price = (current['high'] + current['low']) / 2
    if avg_price > 0:
        daily_range = (current['high'] - current['low']) / avg_price
    else:
        daily_range = 0
    
    # Range Expansion
    if len(data) >= 6:
        recent_ranges = []
        for j in range(5):
            if len(data) > j+1:
                prev_data = data.iloc[-(j+2)]
                prev_avg = (prev_data['high'] + prev_data['low']) / 2
                if prev_avg > 0:
                    rng = (prev_data['high'] - prev_data['low']) / prev_avg
                    recent_ranges.append(rng)
        
        avg_range = np.mean(recent_ranges) if recent_ranges else daily_range
        range_expansion = daily_range / avg_range if avg_range > 0 else 1
    else:
        range_expansion = 1
    
    # Price Efficiency
    price_range = current['high'] - current['low']
    if price_range > 0:
        price_efficiency = abs(current['close'] - current['open']) / price_range
    else:
        price_efficiency = 0
    
    # Efficiency Confirmation
    if len(data) >= 6:
        recent_efficiency = []
        for j in range(5):
            if len(data) > j+1:
                prev_data = data.iloc[-(j+2)]
                prev_range = prev_data['high'] - prev_data['low']
                if prev_range > 0:
                    eff = abs(prev_data['close'] - prev_data['open']) / prev_range
                    recent_efficiency.append(eff)
        
        avg_efficiency = np.mean(recent_efficiency) if recent_efficiency else price_efficiency
        efficiency_momentum = price_efficiency - avg_efficiency
    else:
        efficiency_momentum = 0
    
    # Volume Intensity
    if len(data) >= 20:
        median_volume = data['volume'].iloc[-20:].median()
        volume_intensity = current['volume'] / median_volume if median_volume > 0 else 1
    else:
        volume_intensity = 1
    
    # Intraday Pressure for directional logic
    if price_range > 0:
        intraday_pressure = (current['close'] - current['low']) / price_range - 0.5
    else:
        intraday_pressure = 0
    
    signal = range_expansion * (1 + efficiency_momentum) * volume_intensity * (1 + intraday_pressure)
    return signal
