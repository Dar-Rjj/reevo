import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Compute basic components
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['volatility_proxy'] = (data['high'] - data['low']) / data['open']
    
    # Avoid division by zero
    data['volatility_proxy'] = data['volatility_proxy'].replace(0, np.nan)
    
    # Volume calculations
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Volatility-normalized intraday return with liquidity filter
    data['vol_adj_return'] = data['intraday_return'] / data['volatility_proxy']
    data['liquidity_scale'] = data['volume'] / data['volume_ma_20']
    data['liquidity_scale'] = data['liquidity_scale'].clip(lower=0.1, upper=10)  # Bound extreme values
    
    # Apply sign preservation with magnitude adjustment
    data['base_factor'] = np.sign(data['vol_adj_return']) * np.sqrt(np.abs(data['vol_adj_return'])) * data['liquidity_scale']
    
    # Liquidity acceleration and turnover dynamics
    data['volume_acceleration'] = (data['volume'] - data['volume_ma_5']) / data['volume_ma_5']
    data['turnover_rate'] = data['amount'] / (data['close'] * data['volume'])
    
    # Avoid extreme values in turnover rate
    data['turnover_rate'] = data['turnover_rate'].replace([np.inf, -np.inf], np.nan)
    turnover_quantiles = data['turnover_rate'].quantile([0.01, 0.99])
    data['turnover_rate'] = data['turnover_rate'].clip(lower=turnover_quantiles.iloc[0], upper=turnover_quantiles.iloc[1])
    
    # Multi-factor liquidity momentum with exponential decay
    data['liquidity_momentum'] = data['base_factor'] * data['volume_acceleration'] * data['turnover_rate']
    
    # Apply exponential decay to recent liquidity shocks (5-day half-life)
    decay_weights = np.exp(-np.arange(5) / 5)
    data['liquidity_decayed'] = data['liquidity_momentum'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]) / np.sum(decay_weights[:len(x)]), raw=True
    )
    
    # Range persistence calculations
    data['daily_range'] = data['high'] - data['low']
    
    # 3-day range autocorrelation
    def range_autocorr(x):
        if len(x) < 3:
            return np.nan
        return pd.Series(x).corr(pd.Series(x).shift(1))
    
    data['range_autocorr'] = data['daily_range'].rolling(window=3, min_periods=3).apply(range_autocorr, raw=False)
    
    # Range expansion/contraction regimes
    data['median_range_5'] = data['daily_range'].rolling(window=5, min_periods=3).median()
    data['range_ratio'] = data['daily_range'] / data['median_range_5']
    
    # 3-day price momentum consistency
    data['daily_return'] = data['close'].pct_change()
    data['return_sign'] = np.sign(data['daily_return'])
    
    def momentum_consistency(x):
        if len(x) < 3:
            return np.nan
        signs = np.sign(x)
        return np.sum(signs[1:] == signs[:-1]) / (len(signs) - 1)
    
    data['momentum_consistency'] = data['daily_return'].rolling(window=3, min_periods=3).apply(momentum_consistency, raw=True)
    
    # Final factor construction with regime-based weighting
    # High persistence regime: range_autocorr > 0.3 and range_ratio < 1.2
    high_persistence_mask = (data['range_autocorr'] > 0.3) & (data['range_ratio'] < 1.2)
    
    # Range expansion regime: range_ratio > 1.5
    range_expansion_mask = data['range_ratio'] > 1.5
    
    # Turnover spike regime: volume_acceleration > 0.5
    turnover_spike_mask = data['volume_acceleration'] > 0.5
    
    # Initialize final factor
    data['final_factor'] = data['liquidity_decayed']
    
    # Apply regime-based adjustments
    # Strong reversion signals in high persistence regimes
    data.loc[high_persistence_mask, 'final_factor'] = data['liquidity_decayed'] * (1 + data['range_autocorr'])
    
    # Momentum continuation in range expansion phases
    data.loc[range_expansion_mask, 'final_factor'] = data['liquidity_decayed'] * data['momentum_consistency']
    
    # Liquidity-accelerated adjustments during turnover spikes
    data.loc[turnover_spike_mask, 'final_factor'] = data['liquidity_decayed'] * (1 + data['volume_acceleration'])
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
