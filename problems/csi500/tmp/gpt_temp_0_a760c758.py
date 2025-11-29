import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['abs_gap'] = data['gap'].abs()
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_reversal'] = (data['close'] - data['open']) / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    
    # Reversal efficiency score
    data['reversal_efficiency'] = data['gap'] * data['intraday_reversal']
    
    # Volatility calculations
    data['daily_range_pct'] = data['intraday_range'] / data['prev_close']
    data['volatility_10d'] = data['daily_range_pct'].rolling(window=10, min_periods=5).mean()
    data['close_returns'] = data['close'].pct_change()
    data['volatility_20d'] = data['close_returns'].rolling(window=20, min_periods=10).std()
    
    # Combined volatility measure
    data['combined_vol'] = (data['volatility_10d'] + data['volatility_20d']) / 2
    
    # Volatility regime detection
    vol_median = data['combined_vol'].rolling(window=60, min_periods=30).median()
    data['vol_regime'] = np.where(data['combined_vol'] > vol_median, 1, 0)  # 1 = high vol, 0 = low vol
    
    # Volatility-weighted reversal
    data['vol_weighted_reversal'] = data['reversal_efficiency'] * (1 + data['vol_regime'] * 0.5)
    
    # Multi-timeframe confirmation
    data['reversal_3d_persistence'] = data['reversal_efficiency'].rolling(window=3, min_periods=2).sum()
    data['price_trend_10d'] = data['close'].pct_change(10)
    
    # Range expansion analysis
    data['daily_range_ratio'] = data['intraday_range'] / data['prev_close']
    data['range_5d_avg'] = data['daily_range_ratio'].rolling(window=5, min_periods=3).mean()
    data['range_expansion'] = data['daily_range_ratio'] / data['range_5d_avg']
    
    # Price position efficiency
    data['close_position'] = (data['close'] - data['low']) / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    data['position_strength'] = np.abs(data['close_position'] - 0.5) * 2  # 0-1 scale, extremes = 1
    
    # Volume analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_expansion'] = data['volume'] / data['volume_5d_avg']
    
    # Volume-range correlation (5-day rolling)
    data['volume_range_corr'] = data['daily_range_ratio'].rolling(window=5, min_periods=3).corr(data['volume_expansion'])
    
    # Composite range expansion signal
    data['range_expansion_signal'] = (data['range_expansion'] * data['volume_expansion'] * 
                                     data['position_strength'] * (1 + data['volume_range_corr']))
    
    # Price efficiency metric
    data['price_efficiency'] = np.abs(data['intraday_reversal'])
    data['price_efficiency_3d'] = data['price_efficiency'].rolling(window=3, min_periods=2).mean()
    data['price_efficiency_trend'] = data['price_efficiency'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else 0
    )
    
    # Volume efficiency metric
    data['volume_concentration'] = data['amount'] / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    data['volume_efficiency'] = data['volume_concentration'] / data['volume_concentration'].rolling(window=20, min_periods=10).median()
    
    # Efficiency divergence
    data['efficiency_divergence'] = data['price_efficiency'] - data['volume_efficiency']
    data['divergence_strength'] = data['efficiency_divergence'].abs()
    
    # Amount-based liquidity analysis
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_change'] = data['amount'] / data['amount_5d_avg']
    data['liquidity_concentration'] = data['amount'] / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    
    # Liquidity momentum
    data['amount_trend_3d'] = data['amount'].pct_change(3)
    data['amount_momentum_10d'] = data['amount'].pct_change(10)
    
    # Price-liquidity correlation
    data['price_liquidity_corr'] = data['close_returns'].rolling(window=5, min_periods=3).corr(data['amount_change'])
    
    # Composite liquidity signal
    data['liquidity_signal'] = (data['amount_change'] * data['liquidity_concentration'] * 
                               (1 + data['price_liquidity_corr']))
    
    # Breakout analysis
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['breakout_up'] = (data['close'] > data['prev_high']).astype(int)
    data['breakout_down'] = (data['close'] < data['prev_low']).astype(int)
    data['breakout_signal'] = data['breakout_up'] - data['breakout_down']
    
    # Breakout efficiency
    data['breakout_follow_through'] = data['close'].pct_change(1).shift(-1)  # Note: This is forward-looking, will handle carefully
    # Remove forward-looking component for actual implementation
    data['breakout_efficiency'] = data['breakout_signal'] * data['intraday_reversal']
    
    # Breakout persistence
    data['breakout_streak'] = data['breakout_signal'].groupby(
        (data['breakout_signal'] != data['breakout_signal'].shift(1)).cumsum()
    ).cumcount() + 1
    data['breakout_streak'] = data['breakout_streak'] * np.sign(data['breakout_signal'])
    
    # Combine all factors with appropriate weights
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        # Volatility-adjusted gap reversal (30%)
        vol_factor = 1 + 0.3 * data['vol_regime'].iloc[i]
        reversal_score = data['vol_weighted_reversal'].iloc[i] * vol_factor
        
        # Range expansion momentum (25%)
        range_score = data['range_expansion_signal'].iloc[i]
        
        # Efficiency divergence (20%)
        div_score = data['efficiency_divergence'].iloc[i] * data['divergence_strength'].iloc[i]
        
        # Liquidity momentum (15%)
        liquidity_score = data['liquidity_signal'].iloc[i]
        
        # Breakout persistence (10%)
        breakout_score = data['breakout_efficiency'].iloc[i] * (1 + 0.1 * abs(data['breakout_streak'].iloc[i]))
        
        # Combined factor
        combined_factor = (
            0.30 * reversal_score +
            0.25 * range_score +
            0.20 * div_score +
            0.15 * liquidity_score +
            0.10 * breakout_score
        )
        
        result.iloc[i] = combined_factor
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
