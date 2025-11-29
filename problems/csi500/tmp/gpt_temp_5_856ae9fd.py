import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic metrics
    data['prev_close'] = data['close'].shift(1)
    data['returns'] = data['close'].pct_change()
    
    # Calculate True Range and ATR
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    data['atr_20'] = data['true_range'].rolling(window=20).mean()
    
    # Intraday Price Reversal with Volume Confirmation
    # Maximum intraday price deviation
    data['high_deviation'] = (data['high'] - data['prev_close']) / data['prev_close']
    data['low_deviation'] = (data['low'] - data['prev_close']) / data['prev_close']
    data['max_deviation'] = data[['high_deviation', 'low_deviation']].abs().max(axis=1)
    
    # Reversal strength
    data['close_vs_high'] = (data['close'] - data['high']) / data['high']
    data['close_vs_low'] = (data['close'] - data['low']) / data['low']
    data['reversal_strength'] = np.where(
        data['close'] > data['prev_close'],
        data['close_vs_low'],
        data['close_vs_high']
    )
    data['normalized_reversal'] = data['reversal_strength'] / data['atr_5']
    
    # Volume confirmation
    data['volume_median_20'] = data['volume'].rolling(window=20).median()
    data['volume_surge'] = data['volume'] / data['volume_median_20']
    data['volume_acceleration'] = data['volume'].pct_change(periods=3)
    data['volume_signal'] = data['volume_surge'] * (1 + data['volume_acceleration'])
    
    # Volatility adjustment
    data['atr_ratio'] = data['atr_5'] / data['atr_20']
    data['reversal_factor'] = data['normalized_reversal'] * data['volume_signal'] / data['atr_ratio']
    
    # High-Low Range Breakout Persistence
    data['high_10max'] = data['high'].rolling(window=10).max()
    data['low_10min'] = data['low'].rolling(window=10).min()
    
    # Breakout events
    data['new_high'] = data['high'] >= data['high_10max']
    data['new_low'] = data['low'] <= data['low_10min']
    
    # Consecutive breakouts
    data['high_streak'] = data['new_high'].astype(int).groupby(data.index).cumsum()
    data['low_streak'] = data['new_low'].astype(int).groupby(data.index).cumsum()
    
    # Breakout strength with volume confirmation
    data['high_breakout_strength'] = data['high_streak'] * data['volume_surge']
    data['low_breakout_strength'] = data['low_streak'] * data['volume_surge']
    
    # Daily price range
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['avg_daily_range'] = data['daily_range'].rolling(window=10).mean()
    
    # Net breakout momentum
    data['breakout_momentum'] = (data['high_breakout_strength'] - data['low_breakout_strength']) / data['avg_daily_range']
    
    # Opening Gap Fade with Volume Clustering
    data['gap_percentage'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['avg_gap_size'] = data['gap_percentage'].abs().rolling(window=20).mean()
    data['significant_gap'] = data['gap_percentage'].abs() > (1.5 * data['avg_gap_size'])
    
    # Fade strength
    data['intraday_recovery'] = (data['close'] - data['open']) / data['open']
    data['fade_strength'] = np.where(
        data['gap_percentage'] > 0,
        -data['intraday_recovery'],  # Fade up gap
        data['intraday_recovery']    # Fade down gap
    )
    
    # Volume clustering (simplified - using daily volume patterns)
    data['volume_skew'] = data['volume'].rolling(window=5).skew()
    data['gap_fade_factor'] = data['fade_strength'] * data['significant_gap'] * (1 + data['volume_skew'])
    
    # Price-Volume Divergence Oscillator
    data['price_momentum_3'] = data['close'].pct_change(periods=3)
    data['price_momentum_5'] = data['close'].pct_change(periods=5)
    data['volume_momentum_3'] = data['volume'].pct_change(periods=3)
    data['volume_momentum_5'] = data['volume'].pct_change(periods=5)
    
    # Divergence signals
    data['bullish_divergence'] = ((data['price_momentum_3'] < 0) & (data['volume_momentum_3'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['price_momentum_3'] > 0) & (data['volume_momentum_3'] < 0)).astype(int)
    data['divergence_oscillator'] = data['bullish_divergence'] - data['bearish_divergence']
    
    # Volatility Regime Adjusted Momentum
    data['volatility_20'] = data['returns'].rolling(window=20).std()
    data['volatility_252'] = data['returns'].rolling(window=252).std()
    data['volatility_percentile'] = data['volatility_20'].rolling(window=252).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)), raw=False
    )
    
    # Regime-adaptive momentum
    data['momentum_short'] = data['close'].pct_change(periods=3)
    data['momentum_long'] = data['close'].pct_change(periods=10)
    data['regime_momentum'] = np.where(
        data['volatility_percentile'] == 1,
        data['momentum_short'],
        data['momentum_long']
    )
    
    # Volume-weighted momentum
    data['volume_trend'] = data['volume_momentum_3'].rolling(window=5).mean()
    data['volatility_adjusted_momentum'] = data['regime_momentum'] * (1 + data['volume_trend']) / (1 + data['volatility_20'])
    
    # Combine all factors with equal weighting
    factors = [
        data['reversal_factor'],
        data['breakout_momentum'],
        data['gap_fade_factor'],
        data['divergence_oscillator'],
        data['volatility_adjusted_momentum']
    ]
    
    # Normalize each factor by its rolling z-score (20-day window)
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        factor_mean = factor.rolling(window=20).mean()
        factor_std = factor.rolling(window=20).std()
        normalized_factor = (factor - factor_mean) / factor_std
        combined_factor += normalized_factor.fillna(0)
    
    # Final normalization
    combined_factor = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return combined_factor
