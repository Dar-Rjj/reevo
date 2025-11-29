import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price components
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['volatility_measure'] = (data['high'] - data['low']) / data['open']
    data['high_to_close_return'] = (data['high'] - data['close']) / data['close']
    
    # Medium-Term Momentum Components
    data['close_momentum_5d'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Calculate rolling momentum metrics
    data['close_momentum_5d_ma'] = data['close_momentum_5d'].rolling(window=5, min_periods=3).mean()
    data['volume_momentum_5d_ma'] = data['volume_momentum_5d'].rolling(window=5, min_periods=3).mean()
    
    # Momentum Divergence Detection
    data['momentum_divergence'] = data['intraday_return'] - data['close_momentum_5d_ma']
    
    # Volume-Price Confirmation System
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Abnormal Volume Detection
    data['volume_ratio_20d'] = data['volume'] / data['volume_20d_avg']
    data['volume_ratio_5d'] = data['volume'] / data['volume_5d_avg']
    
    # Volume-Price Correlation (10-day window)
    def rolling_corr(x):
        if len(x) < 5:
            return np.nan
        price_changes = x['close'].pct_change().dropna()
        volume_changes = x['volume'].pct_change().dropna()
        if len(price_changes) < 3 or len(volume_changes) < 3:
            return np.nan
        common_idx = price_changes.index.intersection(volume_changes.index)
        if len(common_idx) < 3:
            return np.nan
        return price_changes.loc[common_idx].corr(volume_changes.loc[common_idx])
    
    # Calculate rolling correlation
    corr_results = []
    for i in range(len(data)):
        if i < 9:
            corr_results.append(np.nan)
            continue
        window_data = data.iloc[i-9:i+1]
        corr_results.append(rolling_corr(window_data))
    
    data['volume_price_corr_10d'] = corr_results
    
    # Volume Breakout Signal
    data['volume_breakout'] = ((data['volume_ratio_20d'] > 1.5) & 
                              (data['volume_ratio_5d'] > 1.2) & 
                              (abs(data['intraday_return']) > 0.02)).astype(int)
    
    # Reversal Signal Generation
    # Calculate rolling ranks for extreme movers
    data['high_to_close_rank'] = data['high_to_close_return'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    
    data['intraday_return_rank'] = data['intraday_return'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    
    # Volume Confirmation for Reversal
    data['volume_confirmation'] = data['volume_ratio_5d'] * data['volume_breakout']
    
    # Generate Reversal Signal
    data['reversal_signal'] = -1 * data['high_to_close_rank'] * data['volume_confirmation']
    
    # Factor Construction & Integration
    # Combine Momentum Divergence with Volume-Confirmed Reversal
    data['momentum_reversal_combined'] = data['momentum_divergence'] * data['reversal_signal']
    
    # Adjust for intraday volatility
    data['volatility_adjusted_signal'] = data['momentum_reversal_combined'] / (data['volatility_measure'] + 0.001)
    
    # Trend Persistence Filter
    data['signal_persistence'] = data['volatility_adjusted_signal'].rolling(window=3, min_periods=2).mean()
    
    # Remove extreme outliers using winsorization
    def winsorize_series(series, limits=(0.05, 0.05)):
        if len(series.dropna()) < 10:
            return series
        lower = series.quantile(limits[0])
        upper = series.quantile(1 - limits[1])
        return series.clip(lower=lower, upper=upper)
    
    # Final signal refinement
    data['final_signal'] = winsorize_series(data['signal_persistence'])
    
    # Validate signal coherence
    valid_mask = (
        (data['volume_20d_avg'].notna()) &
        (data['volume_price_corr_10d'].notna()) &
        (data['momentum_divergence'].notna()) &
        (data['reversal_signal'].notna())
    )
    
    # Generate final alpha factor
    alpha_factor = data['final_signal'].where(valid_mask)
    
    return alpha_factor
