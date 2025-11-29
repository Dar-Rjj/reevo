import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics with Regime-Sensitive Anchoring
    """
    data = df.copy()
    
    # Calculate daily price range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # 1. Multi-Scale Fractal Dimension Calculation
    # Price range fractal dimension (3-day window)
    def price_fractal_dimension(series, window=3):
        """Calculate fractal dimension using box-counting method on price ranges"""
        fractals = []
        for i in range(len(series)):
            if i < window - 1:
                fractals.append(np.nan)
                continue
            window_data = series.iloc[i-window+1:i+1]
            if window_data.isna().any():
                fractals.append(np.nan)
                continue
            # Simple box-counting approximation
            range_max = window_data.max()
            range_min = window_data.min()
            if range_max == range_min:
                fractals.append(1.0)
            else:
                # Calculate approximate fractal dimension
                L = range_max - range_min
                N = len(window_data)
                fractals.append(np.log(N) / np.log(L + 1e-10))
        return pd.Series(fractals, index=series.index)
    
    data['price_fractal_3d'] = price_fractal_dimension(data['daily_range'], window=3)
    
    # Volume clustering fractal dimension (5-day window)
    def volume_fractal_dimension(volume_series, window=5):
        """Calculate fractal dimension for volume clustering"""
        fractals = []
        for i in range(len(volume_series)):
            if i < window - 1:
                fractals.append(np.nan)
                continue
            window_data = volume_series.iloc[i-window+1:i+1]
            if window_data.isna().any():
                fractals.append(np.nan)
                continue
            # Volume clustering fractal using log differences
            log_vol = np.log(window_data + 1e-10)
            vol_diff = log_vol.diff().dropna()
            if len(vol_diff) == 0:
                fractals.append(1.0)
            else:
                # Simple fractal dimension estimation
                vol_range = vol_diff.max() - vol_diff.min()
                if vol_range == 0:
                    fractals.append(1.0)
                else:
                    fractals.append(np.log(len(vol_diff)) / np.log(vol_range + 1e-10))
        return pd.Series(fractals, index=volume_series.index)
    
    data['volume_fractal_5d'] = volume_fractal_dimension(data['volume'], window=5)
    
    # Fractal dimension divergence
    data['fractal_divergence'] = data['price_fractal_3d'] - data['volume_fractal_5d']
    
    # 2. Regime-Dependent Volume Anchoring
    # Identify volatility regimes using rolling standard deviation of daily ranges
    data['volatility_regime'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    
    # High/low volatility regime classification
    vol_median = data['volatility_regime'].rolling(window=20, min_periods=10).median()
    data['high_vol_regime'] = (data['volatility_regime'] > vol_median).astype(int)
    
    # Regime-specific volume-price efficiency
    def regime_volume_efficiency(data, regime_col):
        """Calculate volume-price efficiency for different regimes"""
        efficiency = []
        for i in range(len(data)):
            if i < 1:
                efficiency.append(np.nan)
                continue
            current_regime = data[regime_col].iloc[i]
            # Look back 5 days for same regime periods
            regime_mask = (data[regime_col].iloc[max(0,i-4):i+1] == current_regime)
            if regime_mask.sum() < 2:
                efficiency.append(np.nan)
                continue
            
            regime_data = data.iloc[max(0,i-4):i+1]
            regime_data = regime_data[regime_mask]
            
            if len(regime_data) < 2:
                efficiency.append(np.nan)
                continue
            
            # Volume-price efficiency: correlation between volume changes and absolute returns
            vol_changes = regime_data['volume'].pct_change().dropna()
            abs_returns = np.abs(regime_data['close'].pct_change().dropna())
            
            if len(vol_changes) > 1 and len(abs_returns) > 1:
                min_len = min(len(vol_changes), len(abs_returns))
                corr = np.corrcoef(vol_changes.iloc[:min_len], abs_returns.iloc[:min_len])[0,1]
                efficiency.append(corr if not np.isnan(corr) else 0)
            else:
                efficiency.append(0)
        return pd.Series(efficiency, index=data.index)
    
    data['regime_efficiency'] = regime_volume_efficiency(data, 'high_vol_regime')
    
    # Volume persistence across regimes
    data['volume_persistence'] = data['volume'].pct_change().rolling(window=5).std()
    
    # 3. Combine Fractal and Regime Signals
    # Fractal-regime interaction factor
    data['fractal_regime_interaction'] = data['fractal_divergence'] * data['regime_efficiency']
    
    # Incorporate overnight gap dynamics
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fractal_interaction'] = data['overnight_gap'] * data['price_fractal_3d']
    
    # Session-specific fractal patterns
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10)
    data['session_fractal'] = data['intraday_efficiency'] * data['volume_fractal_5d']
    
    # 4. Generate Final Alpha Factor
    # Regime-weighted transformation
    regime_weight = 1 + 0.5 * data['high_vol_regime']  # Higher weight in high vol regimes
    
    # Fractal consistency adjustment
    fractal_consistency = data['price_fractal_3d'].rolling(window=5).std()
    consistency_weight = 1 / (fractal_consistency + 1e-10)
    
    # Combine all components
    main_factor = (
        data['fractal_regime_interaction'] * 0.4 +
        data['gap_fractal_interaction'] * 0.3 +
        data['session_fractal'] * 0.3
    )
    
    # Apply final transformations
    final_factor = main_factor * regime_weight * consistency_weight
    
    # Handle NaN values
    final_factor = final_factor.fillna(0)
    
    return final_factor
