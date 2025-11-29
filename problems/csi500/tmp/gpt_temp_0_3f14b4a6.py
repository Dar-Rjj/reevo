import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Reversal Detection with Liquidity Flow Analysis
    Combines price reversal signals with liquidity confirmation for alpha generation
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Return Extremes Calculation
    data['intraday_gain'] = data['high'] / data['open'] - 1
    data['intraday_loss'] = data['low'] / data['open'] - 1
    
    # Rolling percentiles for extreme move identification (20-day lookback)
    data['gain_percentile'] = data['intraday_gain'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['loss_percentile'] = data['intraday_loss'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Extreme move flags (top/bottom 20%)
    data['extreme_gain'] = (data['gain_percentile'] > 0.8).astype(int)
    data['extreme_loss'] = (data['loss_percentile'] < 0.2).astype(int)
    
    # 2. Reversal Signals
    # Avoid division by zero
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)
    
    data['close_to_high_retrace'] = (data['high'] - data['close']) / high_low_range
    data['close_to_low_retrace'] = (data['close'] - data['low']) / high_low_range
    
    # Retracement velocity (magnitude of intraday reversal)
    data['retrace_velocity'] = np.where(
        data['extreme_gain'] == 1,
        data['close_to_high_retrace'],
        np.where(
            data['extreme_loss'] == 1,
            data['close_to_low_retrace'],
            0
        )
    )
    
    # Reversal persistence (3-day rolling strength)
    data['reversal_persistence'] = data['retrace_velocity'].rolling(window=3, min_periods=1).mean()
    
    # 3. Volume Distribution Analysis
    # Volume concentration (using price range distribution)
    data['volume_concentration'] = (data['volume'] * (data['high'] - data['low']) / 
                                   (data['high'] - data['low']).rolling(window=5, min_periods=3).mean())
    
    # Volume skew (uneven distribution across price levels)
    data['volume_skew'] = (data['close'] - (data['high'] + data['low']) / 2) * data['volume']
    data['volume_skew'] = data['volume_skew'] / data['volume_skew'].rolling(window=10, min_periods=5).std()
    
    # Buy/Sell pressure imbalance
    price_movement = data['close'] - data['open']
    data['pressure_imbalance'] = np.sign(price_movement) * data['volume']
    data['pressure_imbalance'] = data['pressure_imbalance'] / data['pressure_imbalance'].rolling(window=10, min_periods=5).std()
    
    # Volume acceleration
    data['volume_accel'] = data['volume'].pct_change(periods=1)
    data['volume_accel_smooth'] = data['volume_accel'].rolling(window=5, min_periods=3).mean()
    
    # 4. Amount-Based Liquidity Signals
    # Abnormal amount spikes (z-score based)
    data['amount_zscore'] = (data['amount'] - data['amount'].rolling(window=20, min_periods=10).mean()) / \
                           data['amount'].rolling(window=20, min_periods=10).std()
    
    # Amount concentration (variance)
    data['amount_variance'] = data['amount'].rolling(window=5, min_periods=3).std() / \
                             data['amount'].rolling(window=5, min_periods=3).mean()
    
    # Liquidity provision patterns
    data['liquidity_ratio'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_trend'] = data['liquidity_ratio'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # 5. Market Microstructure Integration
    # Volume-Price Divergence
    price_change = data['close'].pct_change(periods=1)
    volume_change = data['volume'].pct_change(periods=1)
    data['volume_price_divergence'] = price_change - volume_change
    data['volume_price_divergence'] = data['volume_price_divergence'].rolling(window=5, min_periods=3).mean()
    
    # Market Impact Sensitivity
    data['price_elasticity'] = (data['close'].pct_change(periods=1) / 
                               data['volume'].pct_change(periods=1).replace(0, np.nan))
    data['price_elasticity'] = data['price_elasticity'].rolling(window=10, min_periods=5).mean()
    
    # 6. Composite Alpha Factor Generation
    # Base reversal signal
    reversal_signal = data['retrace_velocity'] * data['reversal_persistence']
    
    # Liquidity confirmation components
    volume_confirmation = (data['volume_skew'].fillna(0) * data['pressure_imbalance'].fillna(0))
    amount_confirmation = (data['amount_zscore'].fillna(0) * data['liquidity_trend'].fillna(0))
    
    # Microstructure adjustment
    microstructure_context = data['volume_price_divergence'].fillna(0) * data['price_elasticity'].fillna(0)
    
    # Environment-based weighting
    liquidity_env = data['liquidity_ratio'].rolling(window=10, min_periods=5).rank(pct=True)
    
    # Final composite factor
    high_liquidity_component = reversal_signal * microstructure_context
    low_liquidity_component = reversal_signal * data['retrace_velocity']
    
    composite_factor = np.where(
        liquidity_env > 0.7,  # High liquidity environment
        high_liquidity_component,
        np.where(
            liquidity_env < 0.3,  # Low liquidity environment
            low_liquidity_component,
            reversal_signal * (volume_confirmation + amount_confirmation)  # Normal environment
        )
    )
    
    # Normalize the final factor
    composite_factor = pd.Series(composite_factor, index=data.index)
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=10).mean()) / \
                      composite_factor.rolling(window=20, min_periods=10).std()
    
    return composite_factor
