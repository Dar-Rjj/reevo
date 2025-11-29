import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    # Cross-Sectional Price-Volume Entropy Factor
    def price_volume_entropy_factor(df):
        # Calculate daily returns
        returns = df.groupby(level=1)['close'].pct_change().dropna()
        
        # Compute cross-sectional entropy of returns for each date
        entropy_vals = returns.groupby(level=0).apply(
            lambda x: entropy(np.histogram(x, bins=20, density=True)[0])
        )
        
        # Calculate volume concentration ratio (top 10% / total)
        volume_concentration = df.groupby(level=0)['volume'].apply(
            lambda x: x.nlargest(int(len(x) * 0.1)).sum() / x.sum() if len(x) > 0 else 0
        )
        
        # Calculate 5-day persistence of concentration
        concentration_persistence = volume_concentration.rolling(5).mean()
        
        # Calculate 3-day momentum rank
        momentum = df.groupby(level=1)['close'].pct_change(3)
        momentum_rank = momentum.groupby(level=0).rank(pct=True)
        
        # Combine signals
        signal = np.log1p(entropy_vals * concentration_persistence) * momentum_rank
        return signal
    
    # Intraday Range Persistence Oscillator
    def range_persistence_oscillator(df):
        # Calculate daily range
        daily_range = (df['high'] - df['low']) / df['close']
        
        # Calculate range persistence (autocorrelation over 5 days)
        def range_autocorr(x):
            if len(x) < 5:
                return 0
            return x.autocorr(lag=1)
        
        range_persistence = daily_range.groupby(level=1).rolling(5).apply(
            range_autocorr, raw=False
        ).droplevel(0)
        
        # Calculate breakout probability (proximity to recent high/low)
        recent_high = df.groupby(level=1)['high'].rolling(5).max().droplevel(0)
        recent_low = df.groupby(level=1)['low'].rolling(5).min().droplevel(0)
        
        high_proximity = (df['high'] - recent_low) / (recent_high - recent_low)
        low_proximity = (recent_high - df['low']) / (recent_high - recent_low)
        breakout_prob = (high_proximity + low_proximity) / 2
        
        # Calculate volume acceleration (3-day growth rate)
        volume_growth = df.groupby(level=1)['volume'].pct_change(3)
        
        # Combine signals
        signal = np.tanh(range_persistence * breakout_prob) * volume_growth
        return signal
    
    # Cross-Asset Opening Gap Convergence
    def opening_gap_convergence(df):
        # Calculate opening gap for each stock
        prev_close = df.groupby(level=1)['close'].shift(1)
        opening_gap = (df['open'] - prev_close) / prev_close
        
        # Calculate sector average gap (using first character of ticker as sector proxy)
        def get_sector(ticker):
            return ticker[0] if isinstance(ticker, str) and len(ticker) > 0 else 'A'
        
        sectors = df.index.get_level_values(1).map(get_sector)
        sector_gap = opening_gap.groupby([df.index.get_level_values(0), sectors]).transform('mean')
        
        # Calculate gap deviation
        gap_deviation = opening_gap - sector_gap
        
        # Calculate sector momentum (3-day sector return)
        sector_returns = df.groupby(level=1)['close'].pct_change(3)
        sector_momentum = sector_returns.groupby([df.index.get_level_values(0), sectors]).transform('mean')
        
        # Calculate relative volume strength
        sector_volume = df['volume'].groupby([df.index.get_level_values(0), sectors]).transform('mean')
        volume_ratio = df['volume'] / sector_volume
        
        # Combine signals
        signal = (1 / (1 + np.abs(gap_deviation * sector_momentum))) * volume_ratio
        return signal
    
    # Multi-Timeframe Price Fractality Factor
    def price_fractality_factor(df):
        def hurst_exponent(series, window):
            """Approximate Hurst exponent using rescaled range"""
            if len(series) < window:
                return np.nan
            
            # Calculate log returns
            lrets = np.log(series / series.shift(1)).dropna()
            if len(lrets) < window:
                return np.nan
            
            # Simple R/S approximation
            deviations = lrets - lrets.mean()
            cumulative = deviations.cumsum()
            r = cumulative.max() - cumulative.min()
            s = lrets.std()
            
            return np.log(r / s) / np.log(window) if s > 0 else 0
        
        # Calculate short-term (3-day) fractal dimension
        short_term_hurst = df.groupby(level=1)['close'].rolling(3).apply(
            lambda x: hurst_exponent(pd.Series(x), 3), raw=False
        ).droplevel(0)
        
        # Calculate medium-term (10-day) fractal dimension  
        medium_term_hurst = df.groupby(level=1)['close'].rolling(10).apply(
            lambda x: hurst_exponent(pd.Series(x), 10), raw=False
        ).droplevel(0)
        
        # Calculate fractal dimension ratio and persistence change
        fractal_ratio = short_term_hurst / medium_term_hurst
        persistence_change = (short_term_hurst - medium_term_hurst) / medium_term_hurst
        
        # Calculate volume fractal dimension (simplified)
        volume_hurst = df.groupby(level=1)['volume'].rolling(5).apply(
            lambda x: hurst_exponent(pd.Series(x), 5), raw=False
        ).droplevel(0)
        
        # Combine signals
        signal = fractal_ratio * persistence_change * volume_hurst
        return signal
    
    # Amount-Driven Price Reversal Asymmetry
    def reversal_asymmetry_factor(df):
        # Calculate price reversals
        price_change = df.groupby(level=1)['close'].pct_change()
        
        # Define upward reversals (price was down, now up)
        was_down = price_change.shift(1) < 0
        now_up = price_change > 0
        upward_reversal = was_down & now_up
        
        # Define downward reversals (price was up, now down)  
        was_up = price_change.shift(1) > 0
        now_down = price_change < 0
        downward_reversal = was_up & now_down
        
        # Calculate amount-weighted reversal strength
        upward_strength = (upward_reversal * np.abs(price_change) * df['amount']).fillna(0)
        downward_strength = (downward_reversal * np.abs(price_change) * df['amount']).fillna(0)
        
        # Calculate asymmetry ratio
        asymmetry_ratio = np.log1p(upward_strength) - np.log1p(downward_strength)
        
        # Calculate total reversal magnitude
        total_magnitude = upward_strength + downward_strength
        
        # Calculate volume confirmation (reversal volume vs normal volume)
        normal_volume = df.groupby(level=1)['volume'].rolling(10).mean().droplevel(0)
        volume_intensity = df['volume'] / normal_volume
        
        # Combine signals
        signal = (1 / (1 + np.exp(-asymmetry_ratio * total_magnitude))) * volume_intensity
        return signal
    
    # Calculate all factors
    factor1 = price_volume_entropy_factor(df)
    factor2 = range_persistence_oscillator(df)  
    factor3 = opening_gap_convergence(df)
    factor4 = price_fractality_factor(df)
    factor5 = reversal_asymmetry_factor(df)
    
    # Combine factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0)) / 5
    
    return combined_factor
