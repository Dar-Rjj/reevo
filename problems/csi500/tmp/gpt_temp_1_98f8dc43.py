import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_regime_transition'] = np.nan
    
    # Calculate all components and signals
    for i in range(len(data)):
        if i < 1:  # Skip first row due to lagged variables
            factor.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # 1. Intraday Price Compression Dynamics
        price_compression_ratio = (current['high'] - current['low']) / abs(current['open'] - current['close'])
        price_compression_ratio = np.clip(price_compression_ratio, 0.1, 10)  # Avoid extreme values
        
        volume_compression = current['volume'] / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        compression_efficiency = price_compression_ratio * volume_compression
        
        # Compression signals
        extreme_compression = 1 if price_compression_ratio > 5.0 else 0
        low_compression = 1 if price_compression_ratio < 1.0 else 0
        compression_breakout = current['close'] * compression_efficiency
        
        # 2. Bidirectional Volume Pressure
        upward_pressure_vol = current['volume'] if (current['close'] - current['open']) > 0 else 0
        downward_pressure_vol = current['volume'] if (current['close'] - current['open']) < 0 else 0
        
        pressure_imbalance = (upward_pressure_vol - downward_pressure_vol) / current['volume'] if current['volume'] > 0 else 0
        
        # Pressure momentum (3-day rolling)
        if i >= 3:
            pressure_persistence = data['pressure_imbalance'].iloc[i-2:i+1].sum()
        else:
            pressure_persistence = pressure_imbalance
        
        price_pressure_corr = current['close'] * pressure_imbalance
        pressure_reversal = pressure_imbalance * (current['close'] - current['open'])
        
        # Store pressure imbalance for rolling calculation
        data.loc[data.index[i], 'pressure_imbalance'] = pressure_imbalance
        
        # 3. Session Transition Momentum
        opening_transition = current['open'] / data['prev_close'].iloc[i] if data['prev_close'].iloc[i] > 0 else 1
        intraday_transition = current['close'] / current['open'] if current['open'] > 0 else 1
        transition_divergence = opening_transition - intraday_transition
        
        # Transition persistence (2-day rolling)
        if i >= 2:
            transition_persistence = data['transition_divergence'].iloc[i-1:i+1].mean()
        else:
            transition_persistence = transition_divergence
        
        volume_weighted_transition = transition_divergence * (current['volume'] / current['amount']) if current['amount'] > 0 else 0
        transition_momentum = current['close'] * transition_divergence
        
        # Store transition divergence for rolling calculation
        data.loc[data.index[i], 'transition_divergence'] = transition_divergence
        
        # 4. Price-Level Volume Concentration
        mid_price = (current['high'] + current['low']) / 2
        high_level_concentration = current['volume'] if current['close'] > mid_price else 0
        low_level_concentration = current['volume'] if current['close'] < mid_price else 0
        
        concentration_ratio = high_level_concentration / low_level_concentration if low_level_concentration > 0 else 1
        concentration_ratio = np.clip(concentration_ratio, 0.1, 10)  # Avoid extreme values
        
        concentration_extremes = 1 if concentration_ratio > 3.0 else 0
        concentration_momentum = current['close'] * concentration_ratio
        concentration_shift = high_level_concentration - low_level_concentration
        
        # 5. Range Expansion Efficiency
        prev_range = data['prev_high'].iloc[i] - data['prev_low'].iloc[i]
        range_expansion = (current['high'] - current['low']) / prev_range if prev_range > 0 else 1
        
        volume_expansion = current['volume'] / data['prev_volume'].iloc[i] if data['prev_volume'].iloc[i] > 0 else 1
        
        expansion_efficiency = range_expansion * volume_expansion
        
        high_expansion_cluster = 1 if expansion_efficiency > 2.0 else 0
        low_expansion_cluster = 1 if expansion_efficiency < 0.5 else 0
        expansion_momentum = current['close'] * expansion_efficiency
        
        # 6. Trade Flow Concentration
        positive_flow_volume = current['volume'] if (current['close'] - current['open']) > 0 else 0
        negative_flow_volume = current['volume'] if (current['close'] - current['open']) < 0 else 0
        
        flow_concentration = positive_flow_volume / negative_flow_volume if negative_flow_volume > 0 else 1
        flow_concentration = np.clip(flow_concentration, 0.1, 10)  # Avoid extreme values
        
        # Flow concentration persistence (2-day rolling)
        if i >= 2:
            flow_concentration_persistence = data['flow_concentration'].iloc[i-1:i+1].mean()
        else:
            flow_concentration_persistence = flow_concentration
        
        price_flow_concentration = current['close'] * flow_concentration
        
        # Store flow concentration for rolling calculation
        data.loc[data.index[i], 'flow_concentration'] = flow_concentration
        
        # 7. Volatility Regime Transitions
        volatility_regime = (current['high'] - current['low']) / current['open'] if current['open'] > 0 else 0
        volume_regime = current['volume'] / current['amount'] if current['amount'] > 0 else 0
        regime_transition = volatility_regime * volume_regime
        
        # Store for next iteration
        data.loc[data.index[i], 'prev_regime_transition'] = regime_transition
        
        high_volatility_transition = 1 if i >= 1 and regime_transition > data['prev_regime_transition'].iloc[i-1] else 0
        low_volatility_transition = 1 if i >= 1 and regime_transition < data['prev_regime_transition'].iloc[i-1] else 0
        transition_momentum_vol = current['close'] * regime_transition
        
        # Combine all signals into final factor
        factor_value = (
            compression_breakout * 0.15 +
            pressure_reversal * 0.12 +
            transition_momentum * 0.13 +
            concentration_momentum * 0.14 +
            expansion_momentum * 0.13 +
            price_flow_concentration * 0.16 +
            transition_momentum_vol * 0.17
        )
        
        factor.iloc[i] = factor_value
    
    # Fill NaN values
    factor = factor.fillna(0)
    
    return factor
