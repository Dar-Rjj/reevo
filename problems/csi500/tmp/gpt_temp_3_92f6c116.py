import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum-Volume Divergence with Fractal Microstructure Analysis
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic intraday components
    data['intraday_range'] = data['high'] - data['low']
    data['open_to_close'] = data['close'] - data['open']
    data['is_up_day'] = data['close'] > data['open']
    
    # Fractal Momentum Ratio
    data['fractal_momentum_ratio'] = np.where(
        data['is_up_day'],
        (data['high'] - data['open']) / np.maximum(data['close'] - data['low'], 1e-8),
        (data['open'] - data['low']) / np.maximum(data['high'] - data['close'], 1e-8)
    )
    
    # Effective Spread Proxy
    data['effective_spread'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Volume characteristics
    data['volume_variance_ratio'] = data['volume'].rolling(window=5, min_periods=3).var() / np.maximum(data['volume'].rolling(window=5, min_periods=3).mean(), 1e-8)
    
    # Calculate rolling momentum clustering intensity (simplified)
    def calculate_momentum_clustering(window_data):
        if len(window_data) < 3:
            return 0.0
        price_changes = window_data['close'].diff().dropna()
        reversals = ((price_changes.shift(1) * price_changes) < 0).sum()
        return reversals / len(price_changes) if len(price_changes) > 0 else 0.0
    
    # Calculate momentum clustering using rolling apply
    momentum_clustering = []
    for i in range(len(data)):
        if i < 4:  # Need at least 3 observations for meaningful calculation
            momentum_clustering.append(0.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]  # 5-day window
            clustering = calculate_momentum_clustering(window_data)
            momentum_clustering.append(clustering)
    
    data['momentum_clustering'] = momentum_clustering
    
    # Volume acceleration divergence (simplified)
    def calculate_volume_acceleration(window_data):
        if len(window_data) < 3:
            return 0.0
        price_changes = window_data['close'].diff().dropna()
        volume_data = window_data['volume'].iloc[1:]  # Align with price changes
        
        # Identify acceleration vs deceleration periods
        accel_periods = (price_changes.abs().diff() > 0)
        decel_periods = (price_changes.abs().diff() < 0)
        
        accel_volume = volume_data[accel_periods].sum() if accel_periods.any() else 0
        decel_volume = volume_data[decel_periods].sum() if decel_periods.any() else 0
        total_volume = volume_data.sum()
        
        return (accel_volume - decel_volume) / max(total_volume, 1e-8)
    
    # Calculate volume acceleration divergence
    volume_accel_div = []
    for i in range(len(data)):
        if i < 4:
            volume_accel_div.append(0.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            div = calculate_volume_acceleration(window_data)
            volume_accel_div.append(div)
    
    data['volume_accel_divergence'] = volume_accel_div
    
    # Fractal dimension proxy for price path
    def calculate_price_fractal(window_data):
        if len(window_data) < 3:
            return 1.0
        price_changes = window_data['close'].diff().abs().dropna()
        if len(price_changes) == 0:
            return 1.0
        sum_changes = price_changes.sum()
        if sum_changes <= 0:
            return 1.0
        fractal_dim = np.log(sum_changes) / np.log(len(price_changes))
        return min(max(fractal_dim, 1.0), 2.0)  # Bound between 1 and 2
    
    # Calculate price fractal dimension
    price_fractal = []
    for i in range(len(data)):
        if i < 4:
            price_fractal.append(1.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            fractal = calculate_price_fractal(window_data)
            price_fractal.append(fractal)
    
    data['price_fractal_dim'] = price_fractal
    
    # Volume fractal dimension proxy
    def calculate_volume_fractal(window_data):
        if len(window_data) < 3:
            return 1.0
        volume_changes = window_data['volume'].diff().abs().dropna()
        if len(volume_changes) == 0:
            return 1.0
        sum_changes = volume_changes.sum()
        if sum_changes <= 0:
            return 1.0
        fractal_dim = np.log(sum_changes) / np.log(len(volume_changes))
        return min(max(fractal_dim, 1.0), 2.0)
    
    # Calculate volume fractal dimension
    volume_fractal = []
    for i in range(len(data)):
        if i < 4:
            volume_fractal.append(1.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            fractal = calculate_volume_fractal(window_data)
            volume_fractal.append(fractal)
    
    data['volume_fractal_dim'] = volume_fractal
    
    # Primary divergence components
    data['primary_divergence'] = data['momentum_clustering'] * data['volume_accel_divergence']
    data['fractal_confirmation'] = data['price_fractal_dim'] * data['volume_fractal_dim']
    
    # Spread-momentum interaction (simplified correlation)
    def calculate_spread_momentum(window_data):
        if len(window_data) < 3:
            return 0.0
        spread = window_data['effective_spread']
        momentum = window_data['close'].pct_change().dropna()
        spread_aligned = spread.iloc[1:]  # Align with momentum
        
        if len(momentum) < 2:
            return 0.0
        
        # Simple correlation proxy
        cov = ((spread_aligned - spread_aligned.mean()) * (momentum - momentum.mean())).mean()
        var_spread = spread_aligned.var()
        var_momentum = momentum.var()
        
        if var_spread > 0 and var_momentum > 0:
            return cov / np.sqrt(var_spread * var_momentum)
        return 0.0
    
    # Calculate spread-momentum interaction
    spread_momentum = []
    for i in range(len(data)):
        if i < 4:
            spread_momentum.append(0.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            interaction = calculate_spread_momentum(window_data)
            spread_momentum.append(interaction)
    
    data['spread_momentum_interaction'] = spread_momentum
    
    # Micro-trend persistence (simplified)
    def calculate_micro_trend_persistence(window_data):
        if len(window_data) < 3:
            return 0.0
        price_changes = window_data['close'].diff().dropna()
        if len(price_changes) < 2:
            return 0.0
        
        same_direction = 0
        total_periods = 0
        current_direction = 0
        current_streak = 0
        
        for change in price_changes:
            direction = 1 if change > 0 else (-1 if change < 0 else 0)
            if direction == current_direction and direction != 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    same_direction += current_streak
                    total_periods += 1
                current_direction = direction
                current_streak = 1 if direction != 0 else 0
        
        if current_streak > 0:
            same_direction += current_streak
            total_periods += 1
        
        return same_direction / max(total_periods, 1) if total_periods > 0 else 0.0
    
    # Calculate micro-trend persistence
    micro_trend = []
    for i in range(len(data)):
        if i < 4:
            micro_trend.append(0.0)
        else:
            window_data = data.iloc[max(0, i-4):i+1]
            persistence = calculate_micro_trend_persistence(window_data)
            micro_trend.append(persistence)
    
    data['micro_trend_persistence'] = micro_trend
    
    # Microstructure alignment
    data['microstructure_alignment'] = data['micro_trend_persistence'] * data['spread_momentum_interaction']
    
    # Adaptive smoothing based on fractal complexity
    def adaptive_smooth(series, fractal_dim, min_window=3, max_window=10):
        smoothed = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if pd.isna(series.iloc[i]) or pd.isna(fractal_dim.iloc[i]):
                smoothed.iloc[i] = series.iloc[i]
                continue
            
            # Higher fractal dimension = more noise = longer smoothing window
            window_size = min_window + int((fractal_dim.iloc[i] - 1.0) * (max_window - min_window))
            window_size = max(min_window, min(window_size, max_window))
            
            start_idx = max(0, i - window_size + 1)
            window_data = series.iloc[start_idx:i+1]
            smoothed.iloc[i] = window_data.mean() if len(window_data) > 0 else series.iloc[i]
        
        return smoothed
    
    # Apply adaptive smoothing to key components
    data['primary_divergence_smooth'] = adaptive_smooth(
        data['primary_divergence'], 
        data['price_fractal_dim']
    )
    
    data['fractal_confirmation_smooth'] = adaptive_smooth(
        data['fractal_confirmation'],
        data['volume_fractal_dim']
    )
    
    data['microstructure_alignment_smooth'] = adaptive_smooth(
        data['microstructure_alignment'],
        (data['price_fractal_dim'] + data['volume_fractal_dim']) / 2
    )
    
    # Signal quality assessment
    data['signal_quality'] = (
        data['fractal_confirmation_smooth'].abs() * 
        data['volume_variance_ratio'].rolling(window=5, min_periods=3).mean()
    )
    
    # Noise filtering based on volume and spread
    volume_threshold = data['volume'].rolling(window=10, min_periods=5).quantile(0.3)
    spread_threshold = data['effective_spread'].rolling(window=10, min_periods=5).quantile(0.7)
    
    # Final alpha factor synthesis
    data['raw_alpha'] = (
        data['primary_divergence_smooth'] * 0.4 +
        data['fractal_confirmation_smooth'] * 0.3 +
        data['microstructure_alignment_smooth'] * 0.3
    )
    
    # Apply filters
    valid_signal = (
        (data['volume'] > volume_threshold) &
        (data['effective_spread'] < spread_threshold) &
        (data['signal_quality'] > data['signal_quality'].rolling(window=10, min_periods=5).quantile(0.2))
    )
    
    data['final_alpha'] = data['raw_alpha'] * valid_signal.astype(float)
    
    # Final normalization
    alpha_series = data['final_alpha'].copy()
    
    # Remove any potential future-looking artifacts
    alpha_series = alpha_series.fillna(0.0)
    
    return alpha_series
