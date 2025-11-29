import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining price-volume fractal properties,
    market regime transitions, order flow imbalance, and cross-asset spillover effects.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Volume Fractal Dimensionality
    # Intraday Price Scaling Properties - Hurst exponent approximation
    def hurst_approximation(series, window=20):
        lags = range(2, 6)
        tau = []
        for lag in lags:
            rs = series.rolling(window).apply(lambda x: (x.max() - x.min()) / x.std() if x.std() > 0 else 1, raw=True)
            tau.append(np.log(rs.mean()) if not rs.isna().all() else 0)
        if len(tau) > 1:
            return np.polyfit(np.log(lags), tau, 1)[0]
        return 0
    
    # Volume-Time Multifractal Spectrum - Volume clustering measure
    def volume_clustering(volume, window=10):
        volume_changes = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        return volume_changes.rolling(window).std() / (volume_changes.rolling(window).mean().abs() + 1e-8)
    
    # Price-Volume Correlation at Different Time Scales
    def multi_scale_corr(price, volume, windows=[5, 10, 20]):
        correlations = []
        for window in windows:
            price_ret = price.pct_change().rolling(window).mean()
            volume_ret = volume.pct_change().rolling(window).mean()
            corr = price_ret.rolling(window).corr(volume_ret)
            correlations.append(corr)
        return pd.concat(correlations, axis=1).mean(axis=1)
    
    # 2. Market Regime Transition Signatures
    # Volatility Regime Change Detection
    def volatility_regime_change(close, short_window=5, long_window=20):
        short_vol = close.pct_change().rolling(short_window).std()
        long_vol = close.pct_change().rolling(long_window).std()
        return (short_vol - long_vol) / (long_vol + 1e-8)
    
    # Liquidity Regime Shift Indicators
    def liquidity_regime(volume, amount, window=10):
        volume_trend = volume.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        amount_trend = amount.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        return volume_trend * amount_trend
    
    # Trend-to-Mean-Reversion Transition Patterns
    def trend_reversion_transition(close, trend_window=10, mean_window=20):
        trend_strength = close.rolling(trend_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        mean_reversion = (close - close.rolling(mean_window).mean()) / close.rolling(mean_window).std()
        return trend_strength * mean_reversion
    
    # 3. Order Flow Imbalance Proxies
    # High-Low-Close Position Relative to Range
    def range_position(high, low, close):
        daily_range = high - low
        position = (close - low) / (daily_range + 1e-8)
        return position.rolling(5).mean()
    
    # Volume-Weighted Price Displacement
    def volume_weighted_displacement(open_price, high, low, close, volume):
        typical_price = (high + low + close) / 3
        displacement = (close - open_price) / (high - low + 1e-8)
        return (displacement * volume).rolling(5).mean()
    
    # Intraday Return Skewness Patterns
    def intraday_skewness(open_price, high, low, close, window=10):
        intraday_high_ret = (high - open_price) / open_price
        intraday_low_ret = (low - open_price) / open_price
        intraday_close_ret = (close - open_price) / open_price
        
        skew_measure = (intraday_high_ret + intraday_low_ret - 2 * intraday_close_ret)
        return skew_measure.rolling(window).mean()
    
    # 4. Cross-Asset Spillover Effects (using sector proxies)
    # Sector Relative Strength Momentum
    def relative_strength_momentum(close, sector_window=10, market_window=20):
        sector_returns = close.pct_change().rolling(sector_window).mean()
        market_returns = close.pct_change().rolling(market_window).mean()
        return sector_returns - market_returns
    
    # Inter-market Volatility Transmission
    def volatility_transmission(close, volume, window=15):
        price_vol = close.pct_change().rolling(window).std()
        volume_vol = volume.pct_change().rolling(window).std()
        return price_vol.rolling(5).corr(volume_vol)
    
    # Liquidity Correlation Breakdown Signals
    def liquidity_correlation_breakdown(volume, amount, window=20, corr_window=5):
        volume_trend = volume.rolling(window).mean()
        amount_trend = amount.rolling(window).mean()
        correlation = volume_trend.rolling(corr_window).corr(amount_trend)
        return correlation.pct_change().rolling(5).std()
    
    # Calculate all components
    components = {}
    
    # Price-Volume Fractal Components
    components['hurst_price'] = data.groupby(level=0)['close'].transform(
        lambda x: hurst_approximation(x, window=20)
    )
    components['volume_cluster'] = volume_clustering(data['volume'], window=10)
    components['price_volume_corr'] = multi_scale_corr(data['close'], data['volume'], [5, 10, 20])
    
    # Market Regime Components
    components['vol_regime'] = volatility_regime_change(data['close'], 5, 20)
    components['liquidity_regime'] = liquidity_regime(data['volume'], data['amount'], 10)
    components['trend_reversion'] = trend_reversion_transition(data['close'], 10, 20)
    
    # Order Flow Components
    components['range_pos'] = range_position(data['high'], data['low'], data['close'])
    components['volume_displacement'] = volume_weighted_displacement(
        data['open'], data['high'], data['low'], data['close'], data['volume']
    )
    components['intraday_skew'] = intraday_skewness(
        data['open'], data['high'], data['low'], data['close'], 10
    )
    
    # Cross-Asset Components
    components['relative_strength'] = relative_strength_momentum(data['close'], 10, 20)
    components['vol_transmission'] = volatility_transmission(data['close'], data['volume'], 15)
    components['liquidity_breakdown'] = liquidity_correlation_breakdown(data['volume'], data['amount'], 20, 5)
    
    # Combine components with equal weights
    factor_df = pd.DataFrame(components)
    
    # Normalize each component by cross-sectional z-score
    def cross_sectional_zscore(series):
        return series.groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    normalized_factors = factor_df.apply(cross_sectional_zscore)
    
    # Final factor as weighted combination (equal weights for simplicity)
    final_factor = normalized_factors.mean(axis=1)
    
    # Smooth the factor with a short moving average
    final_factor_smoothed = final_factor.rolling(3).mean()
    
    return final_factor_smoothed
