import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Volatility Regime Momentum Factor
    Combines volatility regime classification, price momentum patterns, 
    volume-volatility relationships, and multi-timeframe momentum integration
    """
    
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. VOLATILITY REGIME CLASSIFICATION
    
    # Historical volatility assessment
    data['daily_range'] = data['high'] - data['low']
    data['vol_5d'] = data['daily_range'].rolling(window=5).std()
    data['vol_20d_median'] = data['daily_range'].rolling(window=20).median()
    
    # Volatility regime classification
    data['vol_regime'] = np.where(data['vol_5d'] > data['vol_20d_median'], 'high', 'low')
    
    # Intraday volatility structure (simplified - using open/close as proxy for am/pm)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['morning_vol_burst'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low']).replace(0, np.nan)
    
    # Volatility breakout detection
    data['vol_percentile_20d'] = data['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2)) if len(x.dropna()) == 20 else np.nan
    )
    data['vol_breakout'] = np.where(data['vol_percentile_20d'] > 0.7, 1, 
                                   np.where(data['vol_percentile_20d'] < 0.3, -1, 0))
    
    # 2. PRICE MOMENTUM UNDER DIFFERENT VOLATILITY
    
    # ATR for normalization
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    
    # High volatility momentum
    data['high_vol_momentum'] = (data['close'] - data['open']) / data['atr_5'].replace(0, np.nan)
    
    # Low volatility breakout
    data['range_compression'] = data['daily_range'].rolling(window=5).std() / data['daily_range'].rolling(window=20).std()
    data['low_vol_breakout'] = (data['close'] - data['open']) / data['daily_range'].replace(0, np.nan) * data['range_compression']
    
    # Price efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 3. VOLUME-VOLATILITY RELATIONSHIP
    
    # Volume concentration
    data['volume_vol_ratio'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume persistence
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x.dropna()) == 5 else np.nan
    )
    
    # Volume efficiency
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['volume_efficiency'] = (data['close'] - data['open']) * data['volume'] / data['atr_5'].replace(0, np.nan)
    
    # 4. MULTI-TIMEFRAME MOMENTUM INTEGRATION
    
    # Short-term momentum
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['intraday_momentum'] = (data['close'] - data['midpoint']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Overnight momentum
    data['overnight_momentum'] = (data['open'] - data['close'].shift(1)) / data['atr_5'].replace(0, np.nan)
    
    # Medium-term regime context
    data['regime_persistence'] = data['vol_regime'].eq(data['vol_regime'].shift(1)).rolling(window=5).mean()
    
    # Cross-regime momentum
    data['momentum_spillover'] = data['high_vol_momentum'].rolling(window=3).std() / data['high_vol_momentum'].rolling(window=10).std()
    
    # 5. FACTOR CONSTRUCTION
    
    # Volatility-regime weighted momentum
    regime_weight = np.where(data['vol_regime'] == 'high', 1.2, 0.8)
    data['regime_weighted_momentum'] = data['intraday_momentum'] * regime_weight
    
    # Volume confirmation
    volume_confirmation = np.where(
        (data['volume_vol_ratio'] > data['volume_vol_ratio'].rolling(window=20).median()) & 
        (data['volume_trend'] > 0), 1, 0.5
    )
    
    # Multi-timeframe synthesis
    short_term_weight = 0.6
    medium_term_weight = 0.4
    
    short_term_component = (
        data['regime_weighted_momentum'] * 0.4 +
        data['overnight_momentum'] * 0.3 +
        data['price_efficiency'] * 0.3
    )
    
    medium_term_component = (
        data['regime_persistence'] * 0.4 +
        data['momentum_spillover'] * 0.3 +
        data['vol_breakout'] * 0.3
    )
    
    # Final factor construction
    base_factor = (
        short_term_component * short_term_weight + 
        medium_term_component * medium_term_weight
    ) * volume_confirmation
    
    # Signal refinement
    # Remove conflicting signals (high volatility with low volume efficiency)
    conflict_filter = ~((data['vol_regime'] == 'high') & (data['volume_efficiency'] < data['volume_efficiency'].rolling(window=20).quantile(0.3)))
    
    # Enhance with volatility breakout confirmation
    breakout_enhancement = np.where(data['vol_breakout'] != 0, 1.2, 1.0)
    
    # Final factor
    final_factor = base_factor * conflict_filter * breakout_enhancement
    
    # Return as pandas Series with same index as input
    return pd.Series(final_factor, index=data.index, name='vol_regime_momentum_factor')
