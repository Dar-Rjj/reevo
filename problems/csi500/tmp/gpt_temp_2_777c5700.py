import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime Identification
    # Short-term volatility (3-day)
    data['hl_ratio'] = (data['high'] - data['low']) / data['prev_close']
    data['tr_ratio'] = data['true_range'] / data['prev_close']
    
    data['short_vol_hl'] = data['hl_ratio'].rolling(window=3, min_periods=2).std()
    data['short_vol_tr'] = data['tr_ratio'].rolling(window=3, min_periods=2).std()
    data['short_vol'] = (data['short_vol_hl'] + data['short_vol_tr']) / 2
    
    # Medium-term volatility (10-day)
    data['medium_vol_hl'] = data['hl_ratio'].rolling(window=10, min_periods=5).std()
    data['medium_vol_tr'] = data['tr_ratio'].rolling(window=10, min_periods=5).std()
    data['medium_vol'] = (data['medium_vol_hl'] + data['medium_vol_tr']) / 2
    
    # Volatility regime classification
    data['vol_ratio'] = data['short_vol'] / data['medium_vol']
    data['high_vol_regime'] = data['vol_ratio'] > 1.2
    
    # Calculate ATR for different periods
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Regime-Adaptive Reversal Signals
    reversal_signals = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        current_data = data.iloc[:i+1].copy()
        current_day = current_data.iloc[-1]
        
        if current_data['high_vol_regime'].iloc[-1]:  # High volatility regime
            # Estimate 2-hour high and low (assuming first 2 hours)
            if len(current_data) >= 3:
                morning_data = current_data.iloc[-3:]
                high_2hour = morning_data['high'].max()
                low_2hour = morning_data['low'].min()
                
                morning_expansion = (high_2hour - current_day['open']) / current_day['open']
                afternoon_compression = (current_day['close'] - low_2hour) / current_day['close']
                
                # Normalize by ATR
                if current_day['atr_5'] > 0:
                    morning_expansion_norm = morning_expansion / current_day['atr_5']
                    afternoon_compression_norm = afternoon_compression / current_day['atr_5']
                    reversal_signal = morning_expansion_norm * afternoon_compression_norm
                else:
                    reversal_signal = 0
            else:
                reversal_signal = 0
                
        else:  # Low volatility regime
            # Price reversal component
            open_close_return = (current_day['close'] - current_day['open']) / current_day['open']
            prev_close_open_return = (current_day['open'] - current_day['prev_close']) / current_day['prev_close']
            
            if current_day['atr_10'] > 0:
                price_reversal = (open_close_return + prev_close_open_return) / (2 * current_day['atr_10'])
            else:
                price_reversal = 0
            
            # Range momentum component
            current_range = current_day['high'] - current_day['low']
            if len(current_data) >= 2:
                prev_range = current_data['high'].iloc[-2] - current_data['low'].iloc[-2]
                if prev_range > 0:
                    range_ratio_prev = current_range / prev_range
                else:
                    range_ratio_prev = 1
                
                if len(current_data) >= 4:
                    avg_3day_range = (current_data['high'].iloc[-4:-1] - current_data['low'].iloc[-4:-1]).mean()
                    if avg_3day_range > 0:
                        range_ratio_avg = current_range / avg_3day_range
                    else:
                        range_ratio_avg = 1
                else:
                    range_ratio_avg = 1
                
                range_momentum = range_ratio_prev * range_ratio_avg
            else:
                range_momentum = 1
            
            reversal_signal = price_reversal * range_momentum
        
        reversal_signals.iloc[i] = reversal_signal
    
    # Apply 3-day moving average to reversal signals
    data['reversal_signal_raw'] = reversal_signals
    data['reversal_signal'] = data['reversal_signal_raw'].rolling(window=3, min_periods=2).mean()
    
    # Volume-Price Efficiency Filter
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_std_20'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_zscore'] = (data['volume'] - data['volume_ma_20']) / data['volume_std_20'].replace(0, 1)
    
    # Volume efficiency components
    volume_efficiency = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        window_data = data.iloc[i-19:i+1].copy()
        current_data = window_data.iloc[-1]
        
        # Volume clustering efficiency
        high_volume_mask = window_data['volume'] > window_data['volume_ma_20']
        if high_volume_mask.any():
            price_changes = window_data['close'].pct_change().fillna(0)
            volume_clustering = (price_changes[high_volume_mask] / window_data['volume'][high_volume_mask]).sum()
        else:
            volume_clustering = 0
        
        # Volume timing efficiency
        if len(window_data) >= 11:
            returns = window_data['close'].pct_change().fillna(0)
            lag_returns = returns.shift(1).fillna(0)
            
            corr_current = returns.iloc[-10:].corr(window_data['volume'].iloc[-10:])
            corr_lag = lag_returns.iloc[-10:].corr(window_data['volume'].iloc[-10:])
            
            volume_timing = corr_current - corr_lag if not (np.isnan(corr_current) or np.isnan(corr_lag)) else 0
        else:
            volume_timing = 0
        
        # Volume percentile ranking
        volume_percentile = (window_data['volume'].rank().iloc[-1] - 1) / (len(window_data) - 1)
        
        # Combine volume efficiency components
        volume_efficiency.iloc[i] = (volume_clustering + volume_timing) * volume_percentile
    
    data['volume_efficiency'] = volume_efficiency.fillna(0)
    
    # Divergence scoring
    data['volume_deviation'] = data['volume_zscore'] * np.sign(data['volume_efficiency'])
    data['signal_volume_alignment'] = data['reversal_signal'] * data['volume_deviation']
    data['volume_filter'] = data['signal_volume_alignment'] * abs(data['volume_efficiency']) * (1 + abs(data['volume_zscore']))
    
    # Final Factor Combination
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pd.isna(data['reversal_signal'].iloc[i]) or pd.isna(data['volume_filter'].iloc[i]):
            final_factor.iloc[i] = 0
            continue
            
        if data['high_vol_regime'].iloc[i]:
            # High volatility: 70% reversal + 30% volume filter
            combined = 0.7 * data['reversal_signal'].iloc[i] + 0.3 * data['volume_filter'].iloc[i]
            momentum_multiplier = 1.5
        else:
            # Low volatility: 50% reversal + 50% volume filter
            combined = 0.5 * data['reversal_signal'].iloc[i] + 0.5 * data['volume_filter'].iloc[i]
            momentum_multiplier = 0.8
        
        final_factor.iloc[i] = combined * momentum_multiplier
    
    # Final smoothing
    result = final_factor.rolling(window=3, min_periods=2).mean()
    
    return result
