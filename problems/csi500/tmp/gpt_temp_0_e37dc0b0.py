import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adaptive Momentum
    # Calculate short-term volatility (5-day std of high-low range)
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['short_vol'] = df['hl_range'].rolling(window=5).std()
    
    # Calculate long-term volatility (20-day std of high-low range)
    df['long_vol'] = df['hl_range'].rolling(window=20).std()
    
    # Calculate volatility ratio and its rate of change
    df['vol_ratio'] = df['short_vol'] / df['long_vol']
    df['vol_ratio_roc'] = df['vol_ratio'] / df['vol_ratio'].shift(1) - 1
    
    # Calculate momentum components
    df['mom_3d'] = df['close'] / df['close'].shift(3) - 1
    df['mom_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Regime-weighted momentum
    high_vol_weight = np.where(df['vol_ratio'] > 1.2, 0.7, 0.3)
    df['regime_momentum'] = high_vol_weight * df['mom_3d'] + (1 - high_vol_weight) * df['mom_10d']
    
    # Volume confirmation
    df['avg_volume_10d'] = df['volume'].rolling(window=10).mean()
    df['volume_surge'] = df['volume'] / df['avg_volume_10d']
    volatility_signal = df['regime_momentum'] * df['volume_surge']
    
    # Intraday Trend Persistence
    # Opening gap momentum
    df['gap_pct'] = (df['open'] / df['close'].shift(1) - 1)
    
    # Intraday trend consistency
    df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['trend_strength'] = np.abs(df['close_to_high'] - 0.5) * 2
    
    # Track consecutive trend days with exponential decay
    df['gap_direction'] = np.sign(df['gap_pct'])
    df['same_direction'] = (df['gap_direction'] == df['gap_direction'].shift(1)).astype(int)
    
    persistence = np.zeros(len(df))
    decay_factor = 0.8
    for i in range(1, len(df)):
        if df['same_direction'].iloc[i] == 1:
            persistence[i] = decay_factor * persistence[i-1] + df['trend_strength'].iloc[i]
        else:
            persistence[i] = df['trend_strength'].iloc[i]
    
    df['persistence_score'] = persistence
    
    # Volume-price alignment
    df['volume_trend'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['price_trend'] = df['close'] / df['close'].rolling(window=5).mean()
    df['alignment_score'] = np.corrcoef(df['volume_trend'].rolling(window=5).mean().fillna(0), 
                                      df['price_trend'].rolling(window=5).mean().fillna(0))[0,1]
    df['alignment_score'] = df['alignment_score'].fillna(0)
    
    trend_signal = df['trend_strength'] * df['persistence_score'] * (1 + df['alignment_score'])
    
    # Price-Volume Divergence Detection
    # Price momentum oscillator
    df['price_mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_mom_20d'] = df['close'] / df['close'].shift(20) - 1
    df['price_osc'] = df['price_mom_5d'] - df['price_mom_20d']
    
    # Volume momentum oscillator
    df['volume_mom_5d'] = df['volume'] / df['volume'].shift(5).rolling(window=5).mean() - 1
    df['volume_mom_20d'] = df['volume'] / df['volume'].shift(20).rolling(window=20).mean() - 1
    df['volume_osc'] = df['volume_mom_5d'] - df['volume_mom_20d']
    
    # Detect divergences
    df['price_lower_low'] = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    df['volume_higher_low'] = (df['volume'] > df['volume'].shift(1)) & (df['volume_osc'] > df['volume_osc'].shift(1))
    df['bullish_div'] = df['price_lower_low'] & df['volume_higher_low']
    
    df['price_higher_high'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
    df['volume_lower_high'] = (df['volume'] < df['volume'].shift(1)) & (df['volume_osc'] < df['volume_osc'].shift(1))
    df['bearish_div'] = df['price_higher_high'] & df['volume_lower_high']
    
    # Divergence strength
    df['div_strength'] = np.where(df['bullish_div'], df['volume_osc'] - df['price_osc'],
                                 np.where(df['bearish_div'], df['price_osc'] - df['volume_osc'], 0))
    
    divergence_signal = df['div_strength']
    
    # Amount-Based Momentum Efficiency
    # Price momentum acceleration
    df['price_mom'] = df['close'] / df['close'].shift(1) - 1
    df['momentum_accel'] = df['price_mom'] - df['price_mom'].shift(1)
    
    # Amount efficiency (amount per price movement)
    df['amount_efficiency'] = df['amount'] / (df['close'] * np.abs(df['price_mom']) + 1e-8)
    
    # Efficiency ratio (current vs historical)
    df['avg_efficiency_10d'] = df['amount_efficiency'].rolling(window=10).mean()
    df['efficiency_ratio'] = df['amount_efficiency'] / df['avg_efficiency_10d']
    
    # Volume-amount alignment
    df['volume_trend_5d'] = df['volume'].rolling(window=5).mean()
    df['amount_trend_5d'] = df['amount'].rolling(window=5).mean()
    df['volume_amount_corr'] = df['volume_trend_5d'].rolling(window=10).corr(df['amount_trend_5d'])
    df['volume_amount_corr'] = df['volume_amount_corr'].fillna(0)
    
    efficiency_signal = df['momentum_accel'] * df['efficiency_ratio'] * (1 + df['volume_amount_corr'])
    
    # Range Compression Breakout Probability
    # Daily range percentile
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_percentile'] = df['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] < np.percentile(x[:-1], 30)).astype(float), raw=False
    )
    
    # Compression duration and intensity
    df['compression_day'] = df['range_percentile'] > 0.5
    compression_count = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['compression_day'].iloc[i]:
            compression_count[i] = compression_count[i-1] + 1
    
    df['compression_intensity'] = compression_count * df['range_percentile']
    
    # Breakout probability based on volume and price position
    df['volume_accumulation'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['breakout_prob'] = (df['volume_accumulation'] * df['price_position'] * 
                          (1 + df['compression_intensity'] / 10))
    
    breakout_signal = df['compression_intensity'] * df['breakout_prob']
    
    # Combine all signals with equal weighting
    combined_signal = (volatility_signal.fillna(0) + 
                      trend_signal.fillna(0) + 
                      divergence_signal.fillna(0) + 
                      efficiency_signal.fillna(0) + 
                      breakout_signal.fillna(0))
    
    return pd.Series(combined_signal, index=df.index)
