import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Factors
    Combines intraday volatility clustering, price-volume divergence, and extreme price reversion dynamics
    """
    data = df.copy()
    
    # Calculate basic metrics
    data['returns'] = data['close'].pct_change()
    data['range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
    
    # 1. Intraday Volatility Clustering
    # Morning vs afternoon volatility ratio (using first 2 hours vs last 2 hours proxy)
    data['intraday_vol_am'] = data['range'].rolling(window=5, min_periods=3).apply(
        lambda x: np.nanmean(x.iloc[:3]) if len(x) >= 3 else np.nan, raw=False
    )
    data['intraday_vol_pm'] = data['range'].rolling(window=5, min_periods=3).apply(
        lambda x: np.nanmean(x.iloc[2:]) if len(x) >= 3 else np.nan, raw=False
    )
    data['vol_am_pm_ratio'] = data['intraday_vol_am'] / data['intraday_vol_pm']
    
    # Consecutive high volatility days count
    high_vol_threshold = data['range'].rolling(window=20).quantile(0.7)
    data['high_vol_flag'] = (data['range'] > high_vol_threshold).astype(int)
    data['consecutive_high_vol'] = data['high_vol_flag'] * (data['high_vol_flag'].groupby(
        (data['high_vol_flag'] != data['high_vol_flag'].shift()).cumsum()
    ).cumcount() + 1)
    
    # Volatility regime transition - sudden expansion
    data['vol_expansion'] = (data['range'] > data['range'].rolling(window=10).mean() * 1.5).astype(int)
    data['vol_contraction'] = (data['range'] < data['range'].rolling(window=10).mean() * 0.7).astype(int)
    
    # 2. Price-Volume Divergence Patterns
    # Low range high volume occurrences
    low_range_threshold = data['range'].rolling(window=20).quantile(0.3)
    high_volume_threshold = data['volume'].rolling(window=20).quantile(0.7)
    data['low_range_high_volume'] = ((data['range'] < low_range_threshold) & 
                                   (data['volume'] > high_volume_threshold)).astype(int)
    
    # Extended consolidation with elevated volume
    data['consolidation_range'] = data['range'].rolling(window=5).mean()
    data['consolidation_vol_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['extended_consolidation'] = ((data['consolidation_range'] < data['range'].rolling(window=20).quantile(0.4)) & 
                                    (data['consolidation_vol_ratio'] > 1.2)).astype(int)
    
    # Absorption volume detection
    data['price_change_abs'] = abs(data['close'].pct_change())
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['absorption_signal'] = ((data['price_change_abs'] < data['price_change_abs'].rolling(window=20).quantile(0.3)) & 
                               (data['volume_zscore'] > 1.5)).astype(int)
    
    # 3. Extreme Price Reversion Dynamics
    # Maximum deviation from VWAP
    data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
    data['max_vwap_deviation'] = data['vwap_deviation'].rolling(window=5).apply(
        lambda x: np.nanmax(abs(x)) if len(x) == 5 else np.nan, raw=False
    )
    
    # Time-weighted price-VWAP divergence
    data['vwap_divergence_persistance'] = (data['vwap_deviation'] * 
                                         data['vwap_deviation'].rolling(window=3).std()).rolling(window=5).mean()
    
    # High rejection wicks frequency
    data['upper_wick_ratio'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['lower_wick_ratio'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['high_rejection_wicks'] = ((data['upper_wick_ratio'] > 0.4) | (data['lower_wick_ratio'] > 0.4)).rolling(window=5).sum()
    
    # Gap fill failure signals
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fill_ratio'] = (data['close'] - data['open']) / abs(data['gap'] * data['close'].shift(1))
    data['gap_fill_failure'] = ((abs(data['gap']) > 0.01) & 
                              (abs(data['gap_fill_ratio']) < 0.3) & 
                              (np.sign(data['gap']) != np.sign(data['returns']))).astype(int)
    
    # Combine factors with appropriate weights
    factors = pd.DataFrame({
        'vol_regime': data['vol_am_pm_ratio'] * 0.15 + data['consecutive_high_vol'] * 0.1,
        'price_volume_div': data['low_range_high_volume'] * 0.2 + data['extended_consolidation'] * 0.15 + data['absorption_signal'] * 0.15,
        'price_reversion': data['max_vwap_deviation'] * -0.1 + data['vwap_divergence_persistance'] * -0.05 + 
                          data['high_rejection_wicks'] * 0.1 + data['gap_fill_failure'] * 0.1
    })
    
    # Final factor combination
    final_factor = factors.sum(axis=1)
    
    # Normalize and handle missing values
    final_factor = (final_factor - final_factor.rolling(window=20).mean()) / final_factor.rolling(window=20).std()
    
    return final_factor
