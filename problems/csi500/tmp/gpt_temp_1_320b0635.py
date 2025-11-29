import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Temporal Asymmetry with Microstructural Regime Detection
    Generates alpha factors based on temporal price-volume relationships and market regime analysis
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume features
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Rolling window for regime detection (minimum 20 days for stability)
    min_periods = 20
    
    for current_date in data.index[min_periods:]:
        current_idx = data.index.get_loc(current_date)
        window_data = data.iloc[:current_idx+1]  # Only past and current data
        
        # 1. Temporal Price-Volume Dislocation Analysis
        # Lead-Lag Relationship Analysis
        if len(window_data) >= 30:
            # Volume-Price Lead Ratio (using rolling correlations)
            vol_price_corr = window_data['volume'].rolling(window=10, min_periods=5).corr(
                window_data['price_change'].shift(1).fillna(0)
            ).iloc[-1]
            
            price_vol_corr = window_data['price_change'].rolling(window=10, min_periods=5).corr(
                window_data['volume'].shift(1).fillna(0)
            ).iloc[-1]
            
            lead_lag_ratio = vol_price_corr - price_vol_corr if not (np.isnan(vol_price_corr) or np.isnan(price_vol_corr)) else 0
        else:
            lead_lag_ratio = 0
        
        # 2. Microstructural State Transitions
        # Market Regime Identification
        recent_data = window_data.tail(20)
        
        # Liquidity State Classification
        volume_ma = recent_data['volume'].mean()
        volume_std = recent_data['volume'].std()
        liquidity_state = 1 if volume_ma > volume_std else -1
        
        # Volatility Regime Detection
        volatility_ma = recent_data['high_low_range'].mean()
        volatility_state = 1 if volatility_ma > recent_data['high_low_range'].median() else -1
        
        # Trend vs Mean-Reversion Identification
        price_trend = recent_data['close'].pct_change().mean()
        trend_state = 1 if abs(price_trend) > 0.001 else -1
        
        # 3. Regime-Specific Dynamics
        regime_score = liquidity_state + volatility_state + trend_state
        
        # Volume concentration analysis
        high_vol_days = len(recent_data[recent_data['volume'] > volume_ma])
        volume_concentration = high_vol_days / len(recent_data)
        
        # Price responsiveness to volume
        if len(recent_data) >= 10:
            price_vol_responsiveness = recent_data['price_change'].corr(recent_data['volume_change'])
            price_vol_responsiveness = 0 if np.isnan(price_vol_responsiveness) else price_vol_responsiveness
        else:
            price_vol_responsiveness = 0
        
        # 4. Asymmetric Signal Generation
        # Temporal Mismatch Exploitation
        if len(window_data) >= 15:
            # Early volume signal detection
            vol_spike_threshold = window_data['volume'].rolling(window=10).mean().iloc[-1] * 1.2
            recent_vol_spikes = len(window_data.tail(5)[window_data.tail(5)['volume'] > vol_spike_threshold])
            early_volume_signal = recent_vol_spikes / 5
        else:
            early_volume_signal = 0
        
        # 5. Regime-Adaptive Signal Processing
        # Adjust signals based on regime
        if regime_score > 1:  # High activity regime
            regime_multiplier = 1.5
        elif regime_score < -1:  # Low activity regime
            regime_multiplier = 0.7
        else:  # Neutral regime
            regime_multiplier = 1.0
        
        # 6. Multi-Regime Factor Construction
        # Combine temporal and regime components
        temporal_component = lead_lag_ratio * early_volume_signal
        regime_component = regime_score * volume_concentration * price_vol_responsiveness
        
        # Final factor value with regime adaptation
        factor_value = (temporal_component + regime_component) * regime_multiplier
        
        # Apply quality control - ensure reasonable bounds
        factor_value = np.clip(factor_value, -2, 2)
        
        factor_values.loc[current_date] = factor_value
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
