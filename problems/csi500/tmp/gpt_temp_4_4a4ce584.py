import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics Alpha Factor
    Combines fractal efficiency and regime transitions in price-volume dynamics
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Rolling window for fractal analysis (21 days ~ 1 month)
    window = 21
    
    for i in range(window, len(data)):
        current_data = data.iloc[i-window:i+1]
        
        # 1. Price Fractal Efficiency Component
        price_efficiency = _calculate_price_fractal_efficiency(current_data)
        
        # 2. Volume Fractal Coupling Component  
        volume_coupling = _calculate_volume_fractal_coupling(current_data)
        
        # 3. Fractal Regime Transition Component
        regime_momentum = _detect_fractal_regime_transitions(current_data)
        
        # Combine components with weights
        combined_factor = (0.4 * price_efficiency + 
                          0.35 * volume_coupling + 
                          0.25 * regime_momentum)
        
        factor.iloc[i] = combined_factor
    
    # Forward fill for initial window where calculation isn't possible
    factor = factor.ffill()
    
    return factor

def _calculate_price_fractal_efficiency(data):
    """Calculate price path fractal efficiency"""
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    
    # Actual price path length (sum of absolute daily changes)
    actual_path = np.sum(np.abs(np.diff(closes)))
    
    # Linear distance (first to last price)
    linear_distance = np.abs(closes[-1] - closes[0])
    
    # Fractal efficiency: linear_distance / actual_path
    # Higher values indicate more efficient (less tortuous) price movement
    if actual_path > 0:
        efficiency = linear_distance / actual_path
    else:
        efficiency = 0
    
    # Directional changes per unit time
    price_changes = np.diff(closes)
    directional_changes = np.sum(np.abs(np.diff(np.sign(price_changes)))) / len(price_changes)
    
    # High-Low range fractal dimension approximation
    daily_ranges = highs - lows
    range_complexity = np.std(daily_ranges) / np.mean(daily_ranges) if np.mean(daily_ranges) > 0 else 0
    
    # Combine components
    price_factor = efficiency * (1 - 0.3 * directional_changes) * (1 - 0.2 * range_complexity)
    
    return price_factor

def _calculate_volume_fractal_coupling(data):
    """Calculate volume-price fractal coupling"""
    volumes = data['volume'].values
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    
    # Volume clustering across time scales
    volume_std = np.std(volumes)
    volume_mean = np.mean(volumes)
    volume_clustering = volume_std / volume_mean if volume_mean > 0 else 0
    
    # Volume-price correlation dimension approximation
    price_changes = np.diff(closes) / closes[:-1]
    volume_changes = np.diff(volumes) / volumes[:-1]
    
    if len(price_changes) > 1 and len(volume_changes) > 1:
        try:
            # Rolling correlation between absolute price changes and volume changes
            corr_coef = np.corrcoef(np.abs(price_changes[-10:]), volume_changes[-10:])[0,1]
            if np.isnan(corr_coef):
                correlation_dim = 0
            else:
                correlation_dim = (corr_coef + 1) / 2  # Normalize to [0,1]
        except:
            correlation_dim = 0
    else:
        correlation_dim = 0
    
    # Volume cascade patterns during price movements
    large_moves = np.where(np.abs(price_changes) > np.std(price_changes))[0]
    if len(large_moves) > 0:
        volume_cascade = np.mean(volume_changes[large_moves]) if len(large_moves) > 0 else 0
    else:
        volume_cascade = 0
    
    # Combine volume fractal components
    volume_factor = (0.5 * (1 - volume_clustering) + 
                    0.3 * correlation_dim + 
                    0.2 * np.tanh(volume_cascade))
    
    return volume_factor

def _detect_fractal_regime_transitions(data):
    """Detect transitions between different fractal regimes"""
    closes = data['close'].values
    volumes = data['volume'].values
    
    # Split data into two halves for regime comparison
    mid_point = len(closes) // 2
    first_half_closes = closes[:mid_point]
    second_half_closes = closes[mid_point:]
    
    first_half_volumes = volumes[:mid_point]
    second_half_volumes = volumes[mid_point:]
    
    # Calculate Hurst-like exponents for both periods
    hurst_first = _approximate_hurst_exponent(first_half_closes)
    hurst_second = _approximate_hurst_exponent(second_half_closes)
    
    # Volume volatility comparison
    vol_vol_first = np.std(first_half_volumes) / np.mean(first_half_volumes) if np.mean(first_half_volumes) > 0 else 0
    vol_vol_second = np.std(second_half_volumes) / np.mean(second_half_volumes) if np.mean(second_half_volumes) > 0 else 0
    
    # Regime transition signal
    hurst_change = hurst_second - hurst_first
    vol_vol_change = vol_vol_second - vol_vol_first
    
    # Positive when transitioning to more efficient/ordered regimes
    regime_momentum = (0.6 * hurst_change + 0.4 * (-vol_vol_change))
    
    return regime_momentum

def _approximate_hurst_exponent(series):
    """Approximate Hurst exponent using R/S analysis"""
    if len(series) < 10:
        return 0.5
    
    try:
        # Simple R/S approximation
        lags = range(2, min(10, len(series)//2))
        tau = []
        
        for lag in lags:
            # Create non-overlapping subseries
            subseries = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(subseries) < 2:
                continue
                
            # Calculate R/S for each subseries
            rs_values = []
            for sub in subseries:
                if len(sub) > 1:
                    mean_sub = np.mean(sub)
                    deviations = sub - mean_sub
                    cumulative_deviations = np.cumsum(deviations)
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    S = np.std(sub)
                    if S > 0:
                        rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            # Fit log(R/S) vs log(lag)
            hurst, _ = linregress(np.log(lags[:len(tau)]), tau)[:2]
            return max(0, min(1, hurst))  # Bound between 0 and 1
        else:
            return 0.5
    except:
        return 0.5
