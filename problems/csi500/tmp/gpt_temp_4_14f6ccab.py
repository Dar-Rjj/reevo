import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day values
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Calculate daily ranges
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['prev_high'] - data['prev_low']
    
    # Initialize factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Skip if any required previous data is missing
        if pd.isna(prev['close']) or pd.isna(prev['high']) or pd.isna(prev['low']) or pd.isna(prev['volume']) or pd.isna(prev['amount']):
            continue
        
        # 1. Intraday Price Elasticity with Volume Momentum
        try:
            # Price Elasticity Patterns
            opening_elasticity = (current['high'] - current['open']) / (current['open'] - current['low']) if (current['open'] - current['low']) != 0 else 0
            closing_elasticity = (current['close'] - current['low']) / (current['high'] - current['close']) if (current['high'] - current['close']) != 0 else 0
            elasticity_divergence = opening_elasticity - closing_elasticity
            
            # Volume Momentum Dynamics
            volume_acceleration = current['volume'] / prev['volume'] if prev['volume'] != 0 else 1
            amount_momentum = current['amount'] / prev['amount'] if prev['amount'] != 0 else 1
            volume_amount_convergence = volume_acceleration * amount_momentum
            
            # Combine with Range Efficiency
            range_efficiency = current['range'] / prev['range'] if prev['range'] != 0 else 1
            factor1 = elasticity_divergence * volume_amount_convergence * range_efficiency
        except:
            factor1 = 0
        
        # 2. Gap Absorption Capacity with Trade Flow Asymmetry
        try:
            # Gap Absorption Dynamics
            gap_absorption_ratio = (current['close'] - current['open']) / (current['open'] - prev['close']) if (current['open'] - prev['close']) != 0 else 0
            intraday_recovery = current['range'] / abs(current['open'] - prev['close']) if abs(current['open'] - prev['close']) != 0 else 0
            absorption_efficiency = gap_absorption_ratio * intraday_recovery
            
            # Trade Flow Patterns
            morning_flow_intensity = current['volume'] * (current['open'] - prev['close'])
            afternoon_flow_persistence = current['amount'] * (current['close'] - current['open'])
            flow_asymmetry = morning_flow_intensity / afternoon_flow_persistence if afternoon_flow_persistence != 0 else 0
            
            # Combine with Volatility Scaling
            volatility_scale = current['range'] / prev['range'] if prev['range'] != 0 else 1
            factor2 = absorption_efficiency * flow_asymmetry / volatility_scale if volatility_scale != 0 else 0
        except:
            factor2 = 0
        
        # 3. Price Momentum Fragmentation with Volume Clustering
        try:
            # Momentum Fragmentation
            opening_momentum_fragment = (current['high'] - current['open']) * current['volume']
            closing_momentum_fragment = (current['close'] - current['low']) * current['amount']
            momentum_fragmentation_ratio = opening_momentum_fragment / closing_momentum_fragment if closing_momentum_fragment != 0 else 0
            
            # Volume Cluster Patterns
            volume_cluster_size = current['volume'] / prev['volume'] if prev['volume'] != 0 else 1
            amount_cluster_density = current['amount'] / prev['amount'] if prev['amount'] != 0 else 1
            cluster_intensity = volume_cluster_size * amount_cluster_density
            
            # Combine with Range Persistence
            range_persistence = current['range'] / prev['range'] if prev['range'] != 0 else 1
            factor3 = momentum_fragmentation_ratio * cluster_intensity * range_persistence
        except:
            factor3 = 0
        
        # 4. Microstructure Pressure with Directional Efficiency
        try:
            # Micro-pressure Patterns
            opening_pressure = (current['high'] - current['open']) * current['volume']
            closing_pressure = (current['close'] - current['low']) * current['amount']
            pressure_differential = opening_pressure - closing_pressure
            
            # Directional Efficiency
            upward_efficiency = (current['close'] - current['open']) * current['volume'] if current['close'] > current['open'] else 0
            downward_efficiency = (current['open'] - current['close']) * current['amount'] if current['close'] < current['open'] else 0
            net_efficiency = upward_efficiency - downward_efficiency
            
            # Combine with Gap Response
            gap_response = (current['open'] - prev['close']) * current['volume']
            factor4 = pressure_differential * net_efficiency * gap_response
        except:
            factor4 = 0
        
        # 5. Volatility Propagation with Trade Intensity
        try:
            # Volatility Propagation
            short_term_volatility = current['range'] / current['open'] if current['open'] != 0 else 0
            medium_term_volatility = prev['range'] / prev['open'] if prev['open'] != 0 else 0
            volatility_propagation = short_term_volatility / medium_term_volatility if medium_term_volatility != 0 else 0
            
            # Trade Intensity Patterns
            volume_intensity = current['volume'] * (current['close'] - current['open'])
            amount_intensity = current['amount'] * current['range']
            trade_intensity = volume_intensity * amount_intensity
            
            # Combine with Momentum Confirmation
            momentum_signal = (current['close'] - prev['close']) * current['volume']
            factor5 = volatility_propagation * trade_intensity * momentum_signal
        except:
            factor5 = 0
        
        # 6. Price Range Elasticity with Flow Concentration
        try:
            # Range Elasticity
            upper_range_elasticity = (current['high'] - current['open']) / (current['open'] - current['low']) if (current['open'] - current['low']) != 0 else 0
            lower_range_elasticity = (current['close'] - current['low']) / (current['high'] - current['close']) if (current['high'] - current['close']) != 0 else 0
            range_elasticity_differential = upper_range_elasticity - lower_range_elasticity
            
            # Flow Concentration
            morning_flow_concentration = current['volume'] * (current['high'] - current['open'])
            afternoon_flow_concentration = current['amount'] * (current['close'] - current['low'])
            flow_concentration_ratio = morning_flow_concentration / afternoon_flow_concentration if afternoon_flow_concentration != 0 else 0
            
            # Combine with Volatility Persistence
            volatility_persistence = current['range'] / prev['range'] if prev['range'] != 0 else 1
            factor6 = range_elasticity_differential * flow_concentration_ratio * volatility_persistence
        except:
            factor6 = 0
        
        # Combine all factors with equal weighting
        factor_values.iloc[i] = (factor1 + factor2 + factor3 + factor4 + factor5 + factor6) / 6
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
