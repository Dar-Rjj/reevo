import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Intraday Momentum
    data['morning_momentum'] = data['high'] - data['open']
    data['afternoon_momentum'] = data['close'] - data['low']
    data['momentum_sum'] = data['morning_momentum'] + data['afternoon_momentum']
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_adj_momentum'] = data['momentum_sum'] / (data['daily_range'] + 1e-8) * data['volume']
    
    # Gap Reversal with Volume Confirmation
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma_5'] + 1e-8)
    data['gap_volume_product'] = data['gap_magnitude'] * data['volume_ratio']
    data['gap_reversal'] = -data['gap_volume_product'] / ((data['high'] - data['low']) + 1e-8)
    
    # Amplitude-Weighted Price Persistence
    data['daily_amplitude'] = (data['high'] - data['low']) / (data['open'] + 1e-8)
    data['close_return'] = data['close'].pct_change()
    data['sign_count'] = 0
    for i in range(2, len(data)):
        signs = np.sign(data['close_return'].iloc[i-2:i+1])
        data.loc[data.index[i], 'sign_count'] = len([s for s in signs if s == signs.iloc[-1]])
    data['amplitude_persistence'] = data['daily_amplitude'] * data['sign_count']
    data['persistence_signal'] = data['amplitude_persistence'] * data['amount']
    
    # Opening Efficiency with Volume Pressure
    data['opening_efficiency'] = (data['open'] - data['prev_close']) / ((data['high'] - data['low']) + 1e-8)
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_ma_20'] + 1e-8)
    data['short_term_momentum'] = data['close'].pct_change(periods=3)
    data['efficiency_pressure'] = data['opening_efficiency'] * data['volume_concentration']
    data['efficiency_signal'] = data['efficiency_pressure'] * data['short_term_momentum'] / (data['close'] + 1e-8)
    
    # Relative Strength with Volume Divergence
    data['sector_return'] = data['close'].pct_change().rolling(window=10, min_periods=1).mean()
    data['stock_return'] = data['close'].pct_change()
    data['relative_strength'] = data['stock_return'] / (data['sector_return'] + 1e-8)
    data['volume_divergence'] = data['volume'].pct_change() - data['sector_return']
    data['strength_divergence'] = data['relative_strength'] * data['volume_divergence']
    data['activity_weighted'] = data['strength_divergence'] * (data['amount'] * data['volume'])
    
    # Combine all signals with equal weights
    signals = [
        data['vol_adj_momentum'],
        data['gap_reversal'],
        data['persistence_signal'],
        data['efficiency_signal'],
        data['activity_weighted']
    ]
    
    # Z-score normalization for each signal
    normalized_signals = []
    for signal in signals:
        signal_mean = signal.rolling(window=20, min_periods=1).mean()
        signal_std = signal.rolling(window=20, min_periods=1).std()
        normalized = (signal - signal_mean) / (signal_std + 1e-8)
        normalized_signals.append(normalized)
    
    # Equal-weighted combination
    combined_signal = sum(normalized_signals) / len(normalized_signals)
    
    return combined_signal
