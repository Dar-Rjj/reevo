import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic intraday metrics
    data['daily_range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    data['open_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Simulate hourly data using intraday patterns (since we only have daily OHLCV)
    # We'll use rolling windows to estimate intraday dynamics
    data['hourly_compression_ratio'] = data['daily_range'] / data['daily_range'].shift(1)
    data['compression_momentum'] = data['hourly_compression_ratio'] / data['hourly_compression_ratio'].shift(1)
    
    # Calculate compression persistence (consecutive compression periods)
    compression_persistence = []
    current_streak = 0
    for i in range(len(data)):
        if i < 2:
            compression_persistence.append(0)
        else:
            if data['hourly_compression_ratio'].iloc[i] < 1.0 and data['hourly_compression_ratio'].iloc[i-1] < 1.0:
                current_streak += 1
            else:
                current_streak = 0
            compression_persistence.append(current_streak)
    data['compression_persistence'] = compression_persistence
    
    # Identify compression extremes relative to rolling average
    data['compression_ma_5'] = data['hourly_compression_ratio'].rolling(window=5, min_periods=3).mean()
    data['compression_deviation'] = data['hourly_compression_ratio'] / data['compression_ma_5']
    
    # Session efficiency dynamics (using rolling windows to simulate intraday)
    data['morning_efficiency'] = (data['high'] - data['open']) / data['daily_range']
    data['afternoon_efficiency'] = (data['close'] - data['low']) / data['daily_range']
    data['session_efficiency_divergence'] = data['morning_efficiency'] - data['afternoon_efficiency']
    
    # Intraday gap absorption
    data['gap_fill_ratio'] = (
        (np.minimum(data['high'], data['prev_close']) - np.maximum(data['low'], data['prev_close'])) / 
        np.abs(data['open_gap'] * data['prev_close'])
    )
    data['gap_fill_ratio'] = np.where(np.abs(data['open_gap']) < 0.001, 0, data['gap_fill_ratio'])
    data['absorption_momentum'] = (data['close'] - data['prev_close']) / data['prev_close']
    
    # Volume distribution patterns (simulating intraday volume concentration)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_concentration'] = data['volume'] / data['volume_ma_5']
    
    # Morning volume concentration (first 30min estimate)
    data['morning_volume_ratio'] = data['volume'].rolling(window=3).apply(
        lambda x: x.iloc[0] / x.sum() if x.sum() > 0 else 0.5, raw=False
    )
    
    # Afternoon volume intensity (last 30min estimate)
    data['afternoon_volume_ratio'] = data['volume'].rolling(window=3).apply(
        lambda x: x.iloc[-1] / x.sum() if x.sum() > 0 else 0.5, raw=False
    )
    
    data['volume_timing_divergence'] = data['morning_volume_ratio'] - data['afternoon_volume_ratio']
    
    # Multi-timeframe momentum integration
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['momentum_cluster'] = data['intraday_return'].rolling(window=3).apply(
        lambda x: sum((x.iloc[i] > 0) == (x.iloc[i-1] > 0) for i in range(1, len(x))), raw=False
    )
    
    data['compression_scaled_momentum'] = data['intraday_return'] / data['hourly_compression_ratio']
    data['volume_confirmed_acceleration'] = data['intraday_return'] * data['volume_timing_divergence']
    
    # Primary compression-efficiency component
    data['compression_weighted_efficiency'] = (
        data['compression_deviation'] * data['session_efficiency_divergence']
    )
    
    data['gap_absorption_efficiency'] = data['gap_fill_ratio'] * data['volume_timing_divergence']
    
    data['compression_momentum_confirmed'] = (
        data['compression_scaled_momentum'] * np.sign(data['session_efficiency_divergence'])
    )
    
    # Dynamic intraday confirmation signals
    data['volume_weighted_compression'] = data['compression_deviation'] * data['volume_concentration']
    
    data['efficiency_persistence'] = data['session_efficiency_divergence'].rolling(window=3).apply(
        lambda x: sum(np.sign(x.iloc[i]) == np.sign(x.iloc[i-1]) for i in range(1, len(x))), raw=False
    )
    
    data['session_transition_alignment'] = (
        np.sign(data['morning_efficiency'] - 0.5) * np.sign(data['afternoon_efficiency'] - 0.5)
    )
    
    # Compression-breakout detection
    data['compression_breakout'] = (
        (data['hourly_compression_ratio'] > data['compression_ma_5'] * 1.2) & 
        (data['intraday_return'].abs() > data['intraday_return'].rolling(window=10).std())
    ).astype(int)
    
    # Generate composite alpha signal
    alpha_components = [
        data['compression_weighted_efficiency'],
        data['gap_absorption_efficiency'],
        data['compression_momentum_confirmed'],
        data['volume_weighted_compression'],
        data['efficiency_persistence'] * data['session_efficiency_divergence'],
        data['session_transition_alignment'] * data['intraday_return'],
        data['compression_breakout'] * data['volume_confirmed_acceleration']
    ]
    
    # Normalize and combine components
    normalized_components = []
    for component in alpha_components:
        if len(component.dropna()) > 0:
            mean_val = component.rolling(window=20, min_periods=10).mean()
            std_val = component.rolling(window=20, min_periods=10).std()
            normalized = (component - mean_val) / std_val
            normalized_components.append(normalized)
    
    # Equal-weighted composite
    if normalized_components:
        composite_alpha = sum(normalized_components) / len(normalized_components)
    else:
        composite_alpha = pd.Series(index=data.index, data=0.0)
    
    # Clean up and return
    composite_alpha = composite_alpha.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return composite_alpha
