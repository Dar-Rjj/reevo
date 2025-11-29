import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import skew, pearsonr

def heuristics_v2(df):
    """
    Intraday Price-Volume Divergence with Microstructural Regime Switching
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic intraday features (assuming daily data)
    data['returns'] = data['close'].pct_change()
    data['abs_returns'] = data['returns'].abs()
    data['range'] = (data['high'] - data['low']) / data['close']
    data['volume_change'] = data['volume'].pct_change()
    
    # Initialize component scores
    fragmentation_scores = []
    participation_scores = []
    accumulation_scores = []
    breakout_scores = []
    deterioration_scores = []
    exhaustion_scores = []
    info_flow_scores = []
    efficiency_scores = []
    
    # Rolling window for calculations (21 days ~ 1 month)
    window = 21
    
    for i in range(len(data)):
        if i < window:
            # Not enough data for reliable calculations
            fragmentation_scores.append(0)
            participation_scores.append(0)
            accumulation_scores.append(0)
            breakout_scores.append(0)
            deterioration_scores.append(0)
            exhaustion_scores.append(0)
            info_flow_scores.append(0)
            efficiency_scores.append(0)
            continue
            
        # Get rolling window data
        window_data = data.iloc[i-window:i+1]
        
        # 1. Microstructural State Detection
        
        # Price Movement Fragmentation
        returns_data = window_data['returns'].dropna()
        if len(returns_data) < 5:
            frag_score = 0
        else:
            # Directional change frequency (simplified for daily data)
            direction_changes = (returns_data * returns_data.shift(1) < 0).sum()
            change_frequency = direction_changes / len(returns_data)
            
            # Return magnitude dispersion
            return_variance = returns_data.var()
            abs_returns = returns_data.abs()
            iqr = abs_returns.quantile(0.75) - abs_returns.quantile(0.25)
            return_skew = skew(returns_data) if len(returns_data) > 2 else 0
            
            magnitude_dispersion = return_variance * (1 + abs(iqr)) * (1 + abs(return_skew))
            
            # Fragmentation acceleration (rate of change)
            if i >= window + 5:
                prev_window = data.iloc[i-window-5:i-4]['returns'].dropna()
                prev_frag = (prev_window * prev_window.shift(1) < 0).sum() / len(prev_window)
                acceleration = change_frequency - prev_frag
            else:
                acceleration = 0
                
            # Average trading range
            avg_range = window_data['range'].mean()
            
            frag_score = change_frequency * magnitude_dispersion * (1 + acceleration) * (1 + avg_range)
        
        # Volume Participation Regimes
        volume_data = window_data['volume'].dropna()
        returns_vol_data = window_data[['returns', 'volume']].dropna()
        
        if len(returns_vol_data) < 3:
            part_score = 0
        else:
            # Volume concentration asymmetry
            up_intervals = returns_vol_data[returns_vol_data['returns'] > 0]
            down_intervals = returns_vol_data[returns_vol_data['returns'] < 0]
            
            if len(up_intervals) > 0 and len(down_intervals) > 0:
                up_volume_ratio = up_intervals['volume'].sum() / (up_intervals['volume'].sum() + down_intervals['volume'].sum())
                concentration_asymmetry = abs(up_volume_ratio - 0.5) * 2
            else:
                concentration_asymmetry = 0
            
            # Volume-return correlation
            try:
                vol_return_corr = pearsonr(returns_vol_data['returns'].abs(), returns_vol_data['volume'])[0]
                if np.isnan(vol_return_corr):
                    vol_return_corr = 0
            except:
                vol_return_corr = 0
            
            # Persistence (autocorrelation of volume)
            volume_persistence = volume_data.autocorr(lag=1) if len(volume_data) > 1 else 0
            if np.isnan(volume_persistence):
                volume_persistence = 0
            
            # Total volume activity
            volume_activity = volume_data.mean() / (volume_data.std() + 1e-8)
            
            part_score = concentration_asymmetry * abs(vol_return_corr) * (1 + volume_persistence) * (1 + volume_activity)
        
        fragmentation_scores.append(frag_score)
        participation_scores.append(part_score)
        
        # 2. Price-Level Accumulation Patterns
        
        # Transaction density clustering (simplified)
        price_levels = window_data['close']
        volume_at_levels = window_data['volume']
        
        # Volume-weighted price imbalance
        current_price = data.iloc[i]['close']
        above_price = window_data[window_data['close'] > current_price]
        below_price = window_data[window_data['close'] < current_price]
        
        if len(above_price) > 0 and len(below_price) > 0:
            volume_imbalance = (above_price['volume'].sum() - below_price['volume'].sum()) / (above_price['volume'].sum() + below_price['volume'].sum())
        else:
            volume_imbalance = 0
        
        # Persistence (how long imbalance persists)
        if i >= window + 3:
            recent_imbalances = []
            for j in range(1, 4):
                idx = i - j
                if idx >= window:
                    prev_data = data.iloc[idx-window:idx+1]
                    prev_price = data.iloc[idx]['close']
                    prev_above = prev_data[prev_data['close'] > prev_price]
                    prev_below = prev_data[prev_data['close'] < prev_price]
                    if len(prev_above) > 0 and len(prev_below) > 0:
                        prev_imbalance = (prev_above['volume'].sum() - prev_below['volume'].sum()) / (prev_above['volume'].sum() + prev_below['volume'].sum())
                        recent_imbalances.append(prev_imbalance)
            
            if len(recent_imbalances) > 0:
                imbalance_persistence = np.mean([abs(im) for im in recent_imbalances])
            else:
                imbalance_persistence = 0
        else:
            imbalance_persistence = 0
        
        # Price range scaling
        price_range = window_data['range'].mean()
        
        accum_score = abs(volume_imbalance) * (1 + imbalance_persistence) * (1 + price_range)
        
        # Breakout readiness
        recent_range = window_data['range'].tail(5).mean()
        avg_range = window_data['range'].mean()
        compression_intensity = max(0, avg_range - recent_range) / (avg_range + 1e-8)
        
        # Volume contraction before potential breakouts
        recent_volume = window_data['volume'].tail(3).mean()
        avg_volume = window_data['volume'].mean()
        volume_contraction = max(0, avg_volume - recent_volume) / (avg_volume + 1e-8)
        
        # Time in compression (simplified)
        low_range_days = (window_data['range'] < window_data['range'].quantile(0.3)).sum()
        compression_time = low_range_days / len(window_data)
        
        breakout_score = compression_intensity * volume_contraction * (1 + compression_time) * (1 + accum_score)
        
        accumulation_scores.append(accum_score)
        breakout_scores.append(breakout_score)
        
        # 3. Momentum Exhaustion
        
        # Return efficiency degradation
        total_movement = window_data['abs_returns'].sum()
        net_movement = abs(window_data['returns'].sum())
        efficiency_ratio = net_movement / (total_movement + 1e-8) if total_movement > 0 else 0
        efficiency_degradation = 1 - efficiency_ratio
        
        # Volume-momentum divergence
        momentum = window_data['returns'].tail(5).sum()
        volume_trend = window_data['volume_change'].tail(5).mean()
        
        if abs(momentum) > 0.01 and abs(volume_trend) > 0.01:
            volume_divergence = abs(momentum * volume_trend) / (abs(momentum) + abs(volume_trend) + 1e-8)
        else:
            volume_divergence = 0
        
        # Acceleration (rate of deterioration)
        if i >= window + 5:
            prev_window = data.iloc[i-window-5:i-4]
            prev_efficiency = abs(prev_window['returns'].sum()) / (prev_window['abs_returns'].sum() + 1e-8)
            acceleration = efficiency_degradation - (1 - prev_efficiency)
        else:
            acceleration = 0
        
        deterioration_score = efficiency_degradation * volume_divergence * (1 + acceleration) * (1 + abs(momentum))
        
        # Exhaustion reversal probability
        # Support breakdown (failed breakouts)
        high_prices = window_data['high']
        recent_high = data.iloc[i]['high']
        resistance_tests = (high_prices > recent_high * 0.99).sum()
        support_breakdown = resistance_tests / len(window_data)
        
        # Momentum extension
        avg_return = window_data['returns'].abs().mean()
        current_extension = abs(momentum) / (avg_return + 1e-8) if avg_return > 0 else 0
        
        # Recent volatility
        recent_vol = window_data['returns'].tail(10).std()
        
        exhaustion_score = support_breakdown * current_extension * (1 + recent_vol)
        
        deterioration_scores.append(deterioration_score)
        exhaustion_scores.append(exhaustion_score)
        
        # 4. Cross-Timeframe Information Flow
        
        # Lead-lag relationships (simplified with different rolling windows)
        if len(returns_data) >= 10:
            short_term_returns = returns_data.tail(5)
            medium_term_returns = returns_data.tail(10)
            
            try:
                lead_lag_corr = pearsonr(short_term_returns.values[:5], medium_term_returns.values[:5])[0]
                if np.isnan(lead_lag_corr):
                    lead_lag_corr = 0
            except:
                lead_lag_corr = 0
        else:
            lead_lag_corr = 0
        
        # Volatility transmission
        short_vol = returns_data.tail(5).std()
        medium_vol = returns_data.tail(10).std()
        
        if medium_vol > 0:
            vol_transmission = short_vol / medium_vol
        else:
            vol_transmission = 0
        
        # Average volatility
        avg_volatility = returns_data.std()
        
        info_flow_score = abs(lead_lag_corr) * vol_transmission * (1 + avg_volatility)
        
        # Market microstructure efficiency
        # Price discovery speed (simplified)
        autocorr_lag1 = returns_data.autocorr(lag=1) if len(returns_data) > 1 else 0
        if np.isnan(autocorr_lag1):
            autocorr_lag1 = 0
        discovery_speed = 1 - abs(autocorr_lag1)
        
        # Noise-to-signal ratio
        noise_component = returns_data.std() / (abs(returns_data.mean()) + 1e-8) if abs(returns_data.mean()) > 0 else 1
        signal_clarity = 1 / (noise_component + 1e-8)
        
        # Market activity level
        activity_level = volume_data.mean() / (volume_data.std() + 1e-8)
        
        efficiency_score = discovery_speed * signal_clarity * (1 + activity_level)
        
        info_flow_scores.append(info_flow_score)
        efficiency_scores.append(efficiency_score)
    
    # Add scores to dataframe
    data['fragmentation'] = fragmentation_scores
    data['participation'] = participation_scores
    data['accumulation'] = accumulation_scores
    data['breakout'] = breakout_scores
    data['deterioration'] = deterioration_scores
    data['exhaustion'] = exhaustion_scores
    data['info_flow'] = info_flow_scores
    data['efficiency'] = efficiency_scores
    
    # Fill NaN values
    data = data.fillna(0)
    
    # 5. Synthesize Adaptive Microstructural Alpha
    
    alpha_scores = []
    
    for i in range(len(data)):
        if i < window:
            alpha_scores.append(0)
            continue
            
        current = data.iloc[i]
        
        # Classify microstructural state
        fragmentation_norm = current['fragmentation'] / (data['fragmentation'].iloc[:i+1].std() + 1e-8)
        participation_norm = current['participation'] / (data['participation'].iloc[:i+1].std() + 1e-8)
        
        # State classification
        if fragmentation_norm > 1 and participation_norm < 0.5:
            state = 'noise_regime'
        elif fragmentation_norm < 0.5 and participation_norm > 1:
            state = 'trend_regime'
        elif abs(fragmentation_norm - participation_norm) < 0.5:
            state = 'transition_regime'
        else:
            state = 'mixed_regime'
        
        # Dynamic signal weights based on state
        if state == 'noise_regime':
            weights = [0.4, 0.1, 0.2, 0.3]  # fragmentation, accumulation, exhaustion, info_flow
        elif state == 'trend_regime':
            weights = [0.1, 0.4, 0.3, 0.2]
        elif state == 'transition_regime':
            weights = [0.2, 0.2, 0.2, 0.4]
        else:  # mixed_regime
            weights = [0.25, 0.25, 0.25, 0.25]
        
        # Combine accumulation and breakout
        accumulation_component = current['accumulation'] * current['breakout']
        
        # Combine deterioration and exhaustion
        exhaustion_component = current['deterioration'] * current['exhaustion']
        
        # Combine info flow and efficiency
        info_flow_component = current['info_flow'] * current['efficiency']
        
        # State-signal congruence
        state_congruence = 1.0
        if state == 'noise_regime' and current['fragmentation'] > data['fragmentation'].iloc[:i+1].mean():
            state_congruence *= 1.2
        if state == 'trend_regime' and current['accumulation'] > data['accumulation'].iloc[:i+1].mean():
            state_congruence *= 1.2
        if state == 'transition_regime' and current['info_flow'] > data['info_flow'].iloc[:i+1].mean():
            state_congruence *= 1.2
        
        # Composite alpha signal
        components = [
            current['fragmentation'],
            accumulation_component,
            exhaustion_component,
            info_flow_component
        ]
        
        weighted_score = sum(w * c for w, c in zip(weights, components))
        final_alpha = weighted_score * state_congruence
        
        alpha_scores.append(final_alpha)
    
    # Create output series
    alpha_series = pd.Series(alpha_scores, index=data.index)
    
    # Normalize the final alpha
    if len(alpha_series) > window:
        rolling_mean = alpha_series.rolling(window=window, min_periods=1).mean()
        rolling_std = alpha_series.rolling(window=window, min_periods=1).std()
        alpha_normalized = (alpha_series - rolling_mean) / (rolling_std + 1e-8)
    else:
        alpha_normalized = alpha_series
    
    return alpha_normalized
