import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def heuristics_v2(df):
    """
    Price-Volume Temporal Asymmetry with Microstructural Regime Detection
    Generates alpha factors based on temporal price-volume relationships and market regime analysis
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume features
    data['price_change'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['volume_change'] = data['volume'].pct_change()
    data['amount_change'] = data['amount'].pct_change()
    
    # 1. Temporal Price-Volume Dislocation
    # Lead-Lag Relationship Analysis
    def calculate_lead_lag_ratio(window_data):
        if len(window_data) < 5:
            return 0
        try:
            # Volume leading price correlation
            vol_lead = pearsonr(window_data['volume'].iloc[:-1], 
                               window_data['price_change'].iloc[1:])[0]
            # Price leading volume correlation  
            price_lead = pearsonr(window_data['price_change'].iloc[:-1],
                                 window_data['volume'].iloc[1:])[0]
            return vol_lead - price_lead if not np.isnan(vol_lead) and not np.isnan(price_lead) else 0
        except:
            return 0
    
    # Rolling lead-lag analysis (20-day window)
    lead_lag_values = []
    for i in range(len(data)):
        if i >= 20:
            window_data = data.iloc[i-19:i+1]
            lead_lag_values.append(calculate_lead_lag_ratio(window_data))
        else:
            lead_lag_values.append(0)
    data['lead_lag_ratio'] = lead_lag_values
    
    # 2. Microstructural State Transitions
    # Market Regime Identification
    def classify_market_regime(window_data):
        """Classify market regime based on volume and volatility characteristics"""
        if len(window_data) < 10:
            return 0
        
        # Liquidity measure (volume stability)
        volume_std = window_data['volume'].std()
        volume_mean = window_data['volume'].mean()
        liquidity_ratio = volume_mean / (volume_std + 1e-8)
        
        # Volatility measure
        volatility = window_data['high_low_range'].mean()
        
        # Combined regime score
        regime_score = liquidity_ratio - volatility * 100
        
        return regime_score
    
    # Volatility regime detection
    def detect_volatility_regime(window_data):
        """Detect high/low volatility periods"""
        if len(window_data) < 10:
            return 0
        
        price_range = window_data['high_low_range'].mean()
        volume_volatility = window_data['volume_change'].std()
        
        # High volatility when both price range and volume volatility are high
        volatility_score = price_range * volume_volatility * 1000
        
        return volatility_score
    
    # Calculate regime indicators with rolling windows
    regime_scores = []
    volatility_scores = []
    
    for i in range(len(data)):
        if i >= 20:
            window_data = data.iloc[i-19:i+1]
            regime_scores.append(classify_market_regime(window_data))
            volatility_scores.append(detect_volatility_regime(window_data))
        else:
            regime_scores.append(0)
            volatility_scores.append(0)
    
    data['regime_score'] = regime_scores
    data['volatility_score'] = volatility_scores
    
    # 3. Asymmetric Signal Generation
    # Temporal Mismatch Exploitation
    def calculate_early_volume_signal(window_data):
        """Detect volume patterns that precede price movements"""
        if len(window_data) < 10:
            return 0
        
        # Volume acceleration before price movement
        volume_accel = window_data['volume_change'].iloc[-5:].mean()
        price_accel = window_data['price_change'].iloc[-3:].mean()
        
        # Signal when volume accelerates before price
        if len(window_data) >= 8:
            early_volume = window_data['volume_change'].iloc[-8:-3].mean()
            late_price = window_data['price_change'].iloc[-3:].mean()
            signal = early_volume - late_price
        else:
            signal = volume_accel - price_accel
            
        return signal
    
    # Calculate early volume signals
    early_signals = []
    for i in range(len(data)):
        if i >= 20:
            window_data = data.iloc[i-19:i+1]
            early_signals.append(calculate_early_volume_signal(window_data))
        else:
            early_signals.append(0)
    data['early_volume_signal'] = early_signals
    
    # 4. Dynamic Framework Adaptation
    # Real-Time Regime Tracking
    def calculate_regime_probability(current_data, historical_data):
        """Estimate probability of current regime state"""
        if len(historical_data) < 30:
            return 0.5
        
        current_volatility = current_data['high_low_range'].iloc[-5:].mean()
        historical_volatility = historical_data['high_low_range'].mean()
        
        current_liquidity = current_data['volume'].iloc[-5:].mean()
        historical_liquidity = historical_data['volume'].mean()
        
        # Probability based on deviation from historical norms
        vol_deviation = abs(current_volatility - historical_volatility) / (historical_volatility + 1e-8)
        liq_deviation = abs(current_liquidity - historical_liquidity) / (historical_liquidity + 1e-8)
        
        # Higher probability when both measures are stable
        regime_prob = 1.0 - (vol_deviation + liq_deviation) / 2.0
        return max(0, min(1, regime_prob))
    
    # Calculate regime probabilities
    regime_probs = []
    for i in range(len(data)):
        if i >= 50:
            current_window = data.iloc[i-4:i+1]  # Recent 5 days
            historical_window = data.iloc[i-49:i-4]  # Previous 45 days
            regime_probs.append(calculate_regime_probability(current_window, historical_window))
        else:
            regime_probs.append(0.5)
    data['regime_probability'] = regime_probs
    
    # 5. Alpha Factor Integration
    # Multi-Regime Factor Construction
    def calculate_composite_factor(row):
        """Combine signals with regime-aware weighting"""
        lead_lag = row['lead_lag_ratio']
        early_signal = row['early_volume_signal']
        regime_prob = row['regime_probability']
        regime_score = row['regime_score']
        
        # Base factor components
        temporal_component = lead_lag * 0.4 + early_signal * 0.3
        
        # Regime adjustment
        regime_adjustment = regime_prob * regime_score * 0.01
        
        # Volatility adjustment (inverse relationship for stability)
        volatility_adjustment = -row['volatility_score'] * 0.1 if 'volatility_score' in row else 0
        
        composite_factor = temporal_component + regime_adjustment + volatility_adjustment
        
        return composite_factor
    
    # Calculate final alpha factor
    alpha_factor = data.apply(calculate_composite_factor, axis=1)
    
    # Clean and normalize the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    # Remove any lookahead bias by ensuring we only use current and past data
    # All calculations above use rolling windows that only include current and historical data
    
    return alpha_factor
