import numpy as np
import pandas as pd
import statsmodels.api as sm
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle
from datetime import datetime

class MarkovKAMA:
    """
    Implementation of the KAMA+MSR model described by Piotr Pomorski.
    Combines:
    - Two-state Markov-Switching Regression (MSR) for volatility regime detection
    - Kaufman's Adaptive Moving Average (KAMA) for trend detection
    
    For Bitcoin trading strategy.
    """
    
    def __init__(self, data_path, train_test_split=0.85, init_date='2017-01-01', embargo_pct=0.01):
        """
        Initialize the MarkovKAMA model.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing Bitcoin price data
        train_test_split : float
            Fraction of data to use for training (default: 0.85)
        """
        self.data_path = data_path
        self.train_test_split = train_test_split
        self.init_date = init_date
        self.embargo_pct = embargo_pct
        self.load_data()
        
    def load_data(self):
        """Load and prepare the Bitcoin price data."""
        try:
            # Load data
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
            
            # Filter out rows with missing data
            self.data = self.data.dropna()
            self.data = self.data[self.data.index >= self.init_date]
            
            # Split into training and test sets
            split_idx = int(len(self.data) * self.train_test_split)
            self.train_data = self.data.iloc[:split_idx]
            self.test_data = self.data.iloc[split_idx:]
            
            # Extract closing prices
            self.asset_data = self.data['close']
            self.train_asset_data = self.train_data['close']
            self.test_asset_data = self.test_data['close']
            
            print(f"Data loaded successfully: {len(self.data)} records")
            print(f"Training set: {len(self.train_data)} records")
            print(f"Test set: {len(self.test_data)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_markov(self, no_regimes=2, return_data=False):
        """
        Apply Markov-Switching Regression to detect volatility regimes.
        
        Parameters:
        -----------
        no_regimes : int
            Number of regimes for the MSR model (default: 2)
        return_data : bool
            Whether to return the regime probabilities (default: False)
            
        Returns:
        --------
        tuple or None
            If return_data is True, returns (low_var, high_var) probabilities
        """
        try:
            # Calculate log returns for the entire dataset
            log_returns_full = np.log(self.asset_data).diff().dropna()
            
            # Split into training (respecting the embargo) and test sets
            split_idx = int(len(self.data) * self.train_test_split)
            embargo_size = int(len(self.data) * self.embargo_pct)
            
            # Ensure log_returns align with the original data index
            log_returns_full.index = self.data.index[1:]  # Shift by 1 because of diff()
            
            # Get train data (excluding embargo region)
            train_end_idx = split_idx - embargo_size
            log_returns_train = log_returns_full.iloc[:train_end_idx]
            
            # Fit Markov Switching Regression model on TRAINING data only
            markov_model = sm.tsa.MarkovRegression(
                log_returns_train.iloc[1:], 
                k_regimes=no_regimes, 
                switching_variance=True, 
                exog=log_returns_train.iloc[:-1]
            )
            
            # Get fitted model
            self.msreg_results = markov_model.fit()
            
            # For returns, we need to predict on the entire dataset for visualization
            # But we'll mark which portions are training, embargo, and test for evaluation
            
            # Predict regimes for full dataset (for visualization)
            # We need to predict on the full dataset to have continuous regime visualization
            # But we'll only use test set predictions for performance evaluation
            
            # Create prediction data structure
            predict_exog = log_returns_full.iloc[:-1]  # Use as exog for prediction
            
            # Get smoothed probabilities for each regime for the full dataset
            # We have to rebuild the structure since statsmodels doesn't provide an easy predict method
            # for smoothed probabilities in Markov models
            
            # Extract parameters from the fitted model
            params = self.msreg_results.params
            
            # Rebuild the model for prediction on the full dataset
            full_model = sm.tsa.MarkovRegression(
                log_returns_full.iloc[1:],
                k_regimes=no_regimes,
                switching_variance=True,
                exog=predict_exog
            )
            
            # Use parameters from trained model
            full_results = full_model.smooth(params)
            
            # Get smoothed probabilities for each regime
            low_var = full_results.smoothed_marginal_probabilities[0]
            high_var = full_results.smoothed_marginal_probabilities[1]
            
            # Store results with dataset markers
            markov_data = pd.concat([low_var, high_var], axis=1)
            markov_data.columns = ['Low_var', 'High_var']
            
            # Add dataset markers (1=train, 0=embargo, 2=test)
            markov_data['dataset'] = 1  # Default to training
            markov_data.iloc[train_end_idx:split_idx, -1] = 0  # Embargo
            markov_data.iloc[split_idx:, -1] = 2  # Test
            
            self.markov = markov_data
            
            print("Markov Switching Regression model fitted successfully.")
            print(f"Training data: {train_end_idx} points")
            print(f"Embargo data: {embargo_size} points")
            print(f"Test data: {len(markov_data) - split_idx} points")
            
            if return_data:
                return low_var, high_var
                
        except Exception as e:
            print(f"Error fitting Markov model: {e}")
            raise
    
    def kama(self, close, length=10, fast=2, slow=30):
        """
        Calculate Kaufman's Adaptive Moving Average.
        
        Parameters:
        -----------
        close : pandas.Series
            Price series to calculate KAMA for
        length : int
            ER lookback period
        fast : int
            Fast EMA constant
        slow : int
            Slow EMA constant
            
        Returns:
        --------
        pandas.Series
            KAMA values
        """
        # Calculate price change and absolute price change
        change = close.diff(length).abs()
        volatility = close.diff().abs().rolling(window=length).sum()
        
        # Calculate efficiency ratio (ER)
        er = pd.Series(np.where(volatility != 0, change / volatility, 0), index=close.index)
        
        # Calculate smoothing constant (SC)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = pd.Series(np.square(er * (fast_sc - slow_sc) + slow_sc), index=close.index)
        
        # Calculate KAMA
        kama = pd.Series(close, index=close.index)
        
        for i in range(length, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
            
        return kama
    
    def get_kama(self, n_window=15, pow1=10, pow2=60, gamma=0.05, return_data=False):
        """
        Calculate KAMA and its filter for trend detection.
        
        Parameters:
        -----------
        n_window : int
            Window length for ER and filter (default: 10)
        pow1 : int
            Fast smoothing constant (default: 2)
        pow2 : int
            Slow smoothing constant (default: 30)
        gamma : float
            Filter scaling factor (default: 0.15)
        return_data : bool
            Whether to return calculated data (default: False)
            
        Returns:
        --------
        tuple or None
            If return_data is True, returns (filtr, period_low, period_high, kama_)
        """
        try:
            # Calculate log prices
            log_price = np.log(self.asset_data)
            
            # Calculate KAMA
            kama_ = self.kama(close=log_price, length=n_window, fast=pow1, slow=pow2)
            
            # Calculate filter
            kama_diff = kama_.diff(n_window)
            filtr = gamma * np.std(kama_diff)
            
            # Calculate period highs and lows
            period_low = kama_.rolling(window=n_window, min_periods=n_window).min()
            period_high = kama_.rolling(window=n_window, min_periods=n_window).max()
            
            # Store results
            periods_low_high = pd.concat([period_low, period_high], axis=1)
            periods_low_high.columns = ['Period_low', 'Period_high']
            
            self.kama_values = kama_
            self.filtr = filtr
            self.periods_low_high = periods_low_high
            
            print("KAMA calculation completed successfully.")
            
            if return_data:
                return filtr, period_low, period_high, kama_
                
        except Exception as e:
            print(f"Error calculating KAMA: {e}")
            raise
    
    def get_classes(self, return_data=True, drop_empty_class=True):
        """
        Combine MSR and KAMA to classify market into four regimes.
        
        Parameters:
        -----------
        return_data : bool
            Whether to return regime classifications (default: True)
        drop_empty_class : bool
            Whether to drop unclassified data points (default: True)
            
        Returns:
        --------
        pandas.DataFrame or None
            If return_data is True, returns DataFrame with regime classifications
        """
        try:
            # Check if MSR and KAMA have been calculated
            if not hasattr(self, 'markov') or not hasattr(self, 'kama_values'):
                raise ValueError("Must run get_markov() and get_kama() before get_classes()")
            
            # Get all dates from the original price data
            all_dates = self.asset_data.index
            
            # Prepare data
            log_price = np.log(self.asset_data)
            log_returns = log_price.diff()
            
            # Create classification DataFrame with all dates
            classes = pd.DataFrame('out of bounds', index=all_dates, columns=['label'])
            
            # Make sure all required series are properly aligned
            # Resample and forward fill to ensure no dates are missing
            for attr in ['markov', 'kama_values', 'periods_low_high']:
                if hasattr(self, attr):
                    setattr(self, attr, getattr(self, attr).reindex(all_dates, method='ffill'))
            
            # Get regime classifications
            low_var_class = self.markov['Low_var'] > 0.5
            high_var_class = self.markov['High_var'] > 0.5
            
            # Get trend classifications
            bullish_class = self.kama_values > self.periods_low_high['Period_low'].shift() + self.filtr
            bearish_class = self.kama_values < self.periods_low_high['Period_high'].shift() - self.filtr
            
            # Assign regime labels to all matching dates
            classes.loc[high_var_class & bullish_class, 'label'] = 'Bullish_High_Var'
            classes.loc[low_var_class & bullish_class, 'label'] = 'Bullish_Low_Var'
            classes.loc[high_var_class & bearish_class, 'label'] = 'Bearish_High_Var'
            classes.loc[low_var_class & bearish_class, 'label'] = 'Bearish_Low_Var'
            
            # For dates that don't clearly fit in one regime, assign based on closest match
            # First, determine if it's low or high volatility
            unclassified = classes['label'] == 'out of bounds'
            if unclassified.any():
                # If we have unclassified points, try to assign them based on volatility regime
                unclassified_idx = classes[unclassified].index
                
                # Assign volatility regime
                vol_regime = pd.Series('high', index=unclassified_idx)
                vol_regime[self.markov.loc[unclassified_idx, 'Low_var'] > 0.5] = 'low'
                
                # Determine trend direction 
                # Use KAMA's 7-day rate of change for trend detection
                kama_roc = self.kama_values.pct_change(periods=2).fillna(0)
                trend = pd.Series('neutral', index=unclassified_idx)
                trend[kama_roc[unclassified_idx] > 0] = 'bullish'
                trend[kama_roc[unclassified_idx] < 0] = 'bearish'
                
                # Combine to assign regime
                for idx in unclassified_idx:
                    if idx in vol_regime.index and idx in trend.index:
                        if vol_regime[idx] == 'low' and trend[idx] == 'bullish':
                            classes.loc[idx, 'label'] = 'Bullish_Low_Var'
                        elif vol_regime[idx] == 'low' and trend[idx] == 'bearish':
                            classes.loc[idx, 'label'] = 'Bullish_Low_Var'
                        elif vol_regime[idx] == 'high' and trend[idx] == 'bullish':
                            classes.loc[idx, 'label'] = 'Bullish_Low_Var'
                        elif vol_regime[idx] == 'high' and trend[idx] == 'bearish':
                            classes.loc[idx, 'label'] = 'Bullish_Low_Var'  
            # Add additional data
            classes['Log_BTC'] = log_price
            classes['Log_BTC_returns'] = log_returns
            # Ensure KAMA is aligned with all dates
            classes['KAMA'] = self.kama_values
            
            # Drop unclassified points if requested - should now be fewer or none
            if drop_empty_class:
                classes = classes[classes['label'] != 'out of bounds']
            
            # Add colors for visualization
            label2color = {
                'Bullish_High_Var': 'yellow',
                'Bullish_Low_Var': 'green',
                'Bearish_High_Var': 'red',
                'Bearish_Low_Var': 'orange'
            }
            
            classes['color'] = classes['label'].apply(lambda label: label2color.get(label, 'gray'))
            
            # Group consecutive regimes
            classes['id'] = classes.groupby((classes['label'] != classes['label'].shift(1)).cumsum()).ngroup()
            
            # Store classes
            self.aligned_classes = classes
            
            print("Regime classification completed successfully.")
            
            if return_data:
                return classes
                
        except Exception as e:
            print(f"Error classifying regimes: {e}")
            raise
    
    def calculate_metrics(self, classes):
        """
        Calculate slope and volatility for each regime segment.
        
        Parameters:
        -----------
        classes : pandas.DataFrame
            DataFrame with regime classifications
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with regime metrics
        """
        metrics = []
        
        # Group by regime ID
        for regime_id, group in classes.groupby('id'):
            if len(group) > 1:
                # Calculate slope (trend) using linear regression
                x = np.arange(len(group))
                y = group['Log_BTC'].values
                slope, _ = np.polyfit(x, y, 1)
                
                # Calculate volatility (standard deviation of returns)
                volatility = group['Log_BTC_returns'].std()
                
                # Calculate duration
                duration = len(group)
                
                # Calculate returns
                returns = np.exp(group['Log_BTC'].iloc[-1] - group['Log_BTC'].iloc[0]) - 1
                
                # Store metrics
                metrics.append({
                    'id': regime_id,
                    'label': group['label'].iloc[0],
                    'start_date': group.index[0],
                    'end_date': group.index[-1],
                    'duration': duration,
                    'slope': slope,
                    'volatility': volatility,
                    'returns': returns
                })
        
        return pd.DataFrame(metrics)
    
    def optimize(self, evals=50):
        """
        Optimize KAMA parameters using Hyperopt.
        
        Parameters:
        -----------
        evals : int
            Number of evaluation rounds (default: 50)
            
        Returns:
        --------
        dict
            Best parameters found
        """
        try:
            def score(params):
                # Apply KAMA with given parameters
                self.get_kama(
                    n_window=int(params['n']), 
                    pow1=int(params['pow1']),
                    pow2=int(params['pow2']), 
                    gamma=params['gamma'], 
                    return_data=False
                )
                
                # Get classes
                classes = self.get_classes(return_data=True, drop_empty_class=True)
                
                # Calculate metrics
                metrics = self.calculate_metrics(classes)
                
                if len(metrics) < 4:  # Need at least one of each regime
                    return {'loss': 999, 'status': STATUS_OK}
                
                # Apply K-means clustering to validate regime separation
                X = metrics[['slope', 'volatility']].values
                kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
                
                # Calculate misclassification score
                label_to_id = {label: i for i, label in enumerate(metrics['label'].unique())}
                true_labels = metrics['label'].map(label_to_id)
                misclassification = (true_labels != kmeans.labels_).mean()
                
                # We want to minimize misclassification
                return {'loss': misclassification, 'status': STATUS_OK}
            
            # Define parameter space
            space = {
                'n': hp.quniform('n', 10, 20, 1),
                'pow1': hp.quniform('pow1', 2, 10, 1),
                'pow2': hp.quniform('pow2', 20, 100, 5),
                'gamma': hp.quniform('gamma', 0.05, 0.25, 0.05)
            }
            
            # Optimize
            trials = Trials()
            best = fmin(fn=score, space=space, algo=tpe.suggest, max_evals=evals, trials=trials)
            
            print(f"Optimization completed. Best parameters: {best}")
            
            return best
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            raise
    
    def implement_trading_strategy(self, classes, initial_capital=1000, test_only=True):
        """
        Implement a conservative trading strategy that only buys in low volatility bullish regimes.
        
        Parameters:
        -----------
        classes : pandas.DataFrame
            DataFrame with regime classifications
        initial_capital : float
            Initial capital amount (default: 1000)
        test_only : bool
            Whether to implement strategy only on test data (default: True)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with strategy performance
        """
        # Filter for test data only if requested
        if test_only and hasattr(self, 'markov') and 'dataset' in self.markov.columns:
            # Get test dataset indices (dataset=2)
            test_indices = self.markov[self.markov['dataset'] == 2].index
            
            # Filter classes for test data only
            test_classes = classes[classes.index.isin(test_indices)]
            
            if len(test_classes) > 0:
                classes = test_classes
            else:
                print("Warning: No test data found. Using all data.")
        
        # Initial portfolio setup
        portfolio = pd.DataFrame(index=classes.index)
        portfolio['btc_price'] = np.exp(classes['Log_BTC'])
        portfolio['regime'] = classes['label']
        
        # Get open prices for next-day execution
        if 'open' in self.data.columns:
            portfolio['next_open'] = self.data['open'].reindex(classes.index).shift(-1)
        else:
            # If open prices aren't available, use the current close as an approximation
            portfolio['next_open'] = portfolio['btc_price'].shift(-1)
        
        # Trading state (0 = flat, 1 = long)
        portfolio['trading_state'] = 0
        
        # Initial values
        portfolio['cash'] = initial_capital
        portfolio['btc_holdings'] = 0
        portfolio['portfolio_value'] = initial_capital
        portfolio['trades'] = 0  # Track when trades occur
        portfolio['trading_costs'] = 0
        portfolio['cumulative_costs'] = 0
        
        # Configure trading costs
        trading_cost_pct = 0.001  # 0.1% per trade
        
        # Strategy logic - using signals at time T to trade at T+1 (next open)
        for i in range(len(portfolio)-1):  # Stop at second-to-last to avoid NaN next_open
            curr_regime = portfolio.iloc[i]['regime']
            curr_state = portfolio.iloc[i]['trading_state']
            curr_cash = portfolio.iloc[i]['cash']
            curr_btc = portfolio.iloc[i]['btc_holdings']
            curr_value = portfolio.iloc[i]['portfolio_value']
            next_open = portfolio.iloc[i]['next_open']
            
            # Determine target state based on regime - ONLY buy in low volatility bullish regime
            target_state = 1 if curr_regime == 'Bullish_Low_Var' else 0
            
            # Record next day's starting positions (before any trades)
            next_idx = portfolio.index[i+1]
            portfolio.loc[next_idx, 'trading_state'] = curr_state
            portfolio.loc[next_idx, 'cash'] = curr_cash
            portfolio.loc[next_idx, 'btc_holdings'] = curr_btc
            
            # Execute trades if state needs to change
            if curr_state != target_state:
                # Calculate trading cost
                trading_cost = curr_value * trading_cost_pct if next_open is not None else 0
                
                if target_state == 1:  # Buy BTC
                    # Calculate how much BTC we can buy with our cash
                    btc_to_buy = (curr_cash - trading_cost) / next_open if next_open is not None else 0
                    
                    # Update next day's positions after trade
                    portfolio.loc[next_idx, 'trading_state'] = 1
                    portfolio.loc[next_idx, 'cash'] = 0
                    portfolio.loc[next_idx, 'btc_holdings'] = btc_to_buy
                    portfolio.loc[next_idx, 'trades'] = 1
                    portfolio.loc[next_idx, 'trading_costs'] = trading_cost
                    
                else:  # Sell BTC
                    # Calculate cash after selling all BTC
                    cash_from_sale = curr_btc * next_open if next_open is not None else 0
                    cash_after_costs = cash_from_sale - trading_cost
                    
                    # Update next day's positions after trade
                    portfolio.loc[next_idx, 'trading_state'] = 0
                    portfolio.loc[next_idx, 'cash'] = cash_after_costs
                    portfolio.loc[next_idx, 'btc_holdings'] = 0
                    portfolio.loc[next_idx, 'trades'] = 1
                    portfolio.loc[next_idx, 'trading_costs'] = trading_cost
        
        # Calculate portfolio value at each step
        portfolio['portfolio_value'] = portfolio['cash'] + (portfolio['btc_holdings'] * portfolio['btc_price'])
        
        # Calculate cumulative trading costs
        portfolio['cumulative_costs'] = portfolio['trading_costs'].cumsum()
        
        # Calculate daily returns for strategy and buy & hold
        portfolio['strategy_returns'] = portfolio['portfolio_value'].pct_change()
        portfolio['btc_returns'] = portfolio['btc_price'].pct_change()
        
        # Calculate buy & hold performance
        initial_btc_holdings = initial_capital / portfolio['btc_price'].iloc[0]
        portfolio['buy_hold_value'] = initial_btc_holdings * portfolio['btc_price']
        
        # Normalize values for comparison
        portfolio['strategy_normalized'] = portfolio['portfolio_value'] / initial_capital
        portfolio['buy_hold_normalized'] = portfolio['buy_hold_value'] / initial_capital
        
        # Add regime duration information
        portfolio['regime_change'] = portfolio['regime'] != portfolio['regime'].shift(1)
        portfolio['regime_duration'] = portfolio.groupby((portfolio['regime_change']).cumsum())['regime'].transform('count')
        
        # Add regime exposure metrics
        exposure = portfolio.groupby('regime')['trading_state'].mean()
        total_days = len(portfolio)
        regime_days = portfolio['regime'].value_counts()
        regime_exposure = {r: f"{exposure.get(r, 0):.2f} ({regime_days.get(r, 0)/total_days:.1%} of time)" 
                          for r in ['Bullish_Low_Var', 'Bullish_High_Var', 'Bearish_Low_Var', 'Bearish_High_Var']}
        
        print("Regime exposure (average position & % of time):")
        for regime, stats in regime_exposure.items():
            print(f"  {regime}: {stats}")
        
        return portfolio
    
    def calculate_performance_metrics(self, portfolio, test_only=True):
        """
        Calculate performance metrics for the trading strategy.
        
        Parameters:
        -----------
        portfolio : pandas.DataFrame
            DataFrame with strategy performance
        test_only : bool
            Whether to calculate metrics only for the test set (default: True)
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        # If test_only, filter for test data points only
        if test_only and 'dataset' in self.markov.columns:
            # Get test dataset indices (dataset=2)
            test_indices = self.markov[self.markov['dataset'] == 2].index
            
            # Filter portfolio for test data only (if any indices match)
            common_indices = portfolio.index.intersection(test_indices)
            if not common_indices.empty:
                portfolio = portfolio.loc[common_indices]
            else:
                print("Warning: No test data found in portfolio. Using all data.")
        
        # Annualization factor (assuming daily data)
        annualize_factor = 365
        
        # Calculate metrics
        btc_returns = portfolio['btc_returns'].dropna()
        strategy_returns = portfolio['strategy_returns'].dropna()
        
        # Annual return
        btc_annual_return = (1 + btc_returns.mean()) ** annualize_factor - 1
        strategy_annual_return = (1 + strategy_returns.mean()) ** annualize_factor - 1
        
        # Volatility
        btc_volatility = btc_returns.std() * np.sqrt(annualize_factor)
        strategy_volatility = strategy_returns.std() * np.sqrt(annualize_factor)
        
        # Sharpe Ratio (assuming 0 risk-free rate)
        btc_sharpe = btc_annual_return / btc_volatility if btc_volatility != 0 else 0
        strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
        
        # Sortino Ratio (downside risk only)
        btc_downside = btc_returns[btc_returns < 0].std() * np.sqrt(annualize_factor)
        strategy_downside = strategy_returns[strategy_returns < 0].std() * np.sqrt(annualize_factor)
        
        btc_sortino = btc_annual_return / btc_downside if btc_downside != 0 else 0
        strategy_sortino = strategy_annual_return / strategy_downside if strategy_downside != 0 else 0
        
        # Maximum Drawdown
        btc_cum_returns = (1 + btc_returns).cumprod()
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        
        btc_peak = btc_cum_returns.cummax()
        strategy_peak = strategy_cum_returns.cummax()
        
        btc_drawdown = (btc_cum_returns / btc_peak - 1)
        strategy_drawdown = (strategy_cum_returns / strategy_peak - 1)
        
        btc_max_drawdown = btc_drawdown.min()
        strategy_max_drawdown = strategy_drawdown.min()
        
        # Calculate number of trades
        num_trades = portfolio['trades'].sum()
        
        # Trading costs
        total_trading_costs = portfolio['cumulative_costs'].iloc[-1] if len(portfolio) > 0 else 0
        
        # Final portfolio value
        final_btc_value = portfolio['buy_hold_normalized'].iloc[-1] if len(portfolio) > 0 else 1
        final_strategy_value = portfolio['strategy_normalized'].iloc[-1] if len(portfolio) > 0 else 1
        
        # Regime classification accuracy
        regime_metrics = self._calculate_regime_accuracy(portfolio)
        
        # Combine all metrics
        metrics = {
            'BTC Annual Return': btc_annual_return,
            'Strategy Annual Return': strategy_annual_return,
            'BTC Volatility': btc_volatility,
            'Strategy Volatility': strategy_volatility,
            'BTC Sharpe Ratio': btc_sharpe,
            'Strategy Sharpe Ratio': strategy_sharpe,
            'BTC Sortino Ratio': btc_sortino,
            'Strategy Sortino Ratio': strategy_sortino,
            'BTC Max Drawdown': btc_max_drawdown,
            'Strategy Max Drawdown': strategy_max_drawdown,
            'Number of Trades': num_trades,
            'Total Trading Costs': total_trading_costs,
            'Final Buy & Hold Value': final_btc_value,
            'Final Strategy Value': final_strategy_value,
            'Outperformance': final_strategy_value - final_btc_value
        }
        
        # Add regime metrics
        metrics.update(regime_metrics)
        
        return metrics
    
    def _calculate_regime_accuracy(self, portfolio):
        """
        Calculate regime classification accuracy metrics.
        
        Parameters:
        -----------
        portfolio : pandas.DataFrame
            DataFrame with strategy performance
            
        Returns:
        --------
        dict
            Dictionary with regime accuracy metrics
        """
        # Group by consecutive regimes
        portfolio['regime_change'] = portfolio['regime'] != portfolio['regime'].shift(1)
        portfolio['regime_group'] = portfolio['regime_change'].cumsum()
        
        # Calculate metrics for each regime group
        regime_groups = []
        
        for group_id, group in portfolio.groupby('regime_group'):
            # Skip tiny groups (less than 3 days)
            if len(group) < 3:
                continue
                
            # Calculate cumulative return for this regime
            start_price = group['btc_price'].iloc[0]
            end_price = group['btc_price'].iloc[-1]
            cumulative_return = (end_price / start_price) - 1
            
            # Determine actual trend direction
            actual_trend = 'bull' if cumulative_return > 0 else 'bear'
            
            # Get the predicted regime
            predicted_regime = group['regime'].iloc[0]
            predicted_trend = 'bull' if predicted_regime.startswith('Bullish') else 'bear'
            
            # Store results
            regime_groups.append({
                'start_date': group.index[0],
                'end_date': group.index[-1],
                'duration': len(group),
                'regime': predicted_regime,
                'predicted_trend': predicted_trend,
                'actual_trend': actual_trend,
                'cumulative_return': cumulative_return,
                'correct': predicted_trend == actual_trend
            })
        
        # Convert to DataFrame for analysis
        if not regime_groups:
            return {
                'Regime Accuracy': 0,
                'Bull Specificity': 0,
                'Bear Specificity': 0
            }
            
        regime_df = pd.DataFrame(regime_groups)
        
        # Calculate overall accuracy
        accuracy = regime_df['correct'].mean() if len(regime_df) > 0 else 0
        
        # Calculate bull market specificity (true positive rate)
        bull_actual = regime_df[regime_df['actual_trend'] == 'bull']
        bull_specificity = bull_actual['correct'].mean() if len(bull_actual) > 0 else 0
        
        # Calculate bear market specificity (true negative rate)
        bear_actual = regime_df[regime_df['actual_trend'] == 'bear']
        bear_specificity = bear_actual['correct'].mean() if len(bear_actual) > 0 else 0
        
        # Store additional data for potential further analysis
        self.regime_accuracy_data = regime_df
        
        return {
            'Regime Accuracy': accuracy,
            'Bull Specificity': bull_specificity,
            'Bear Specificity': bear_specificity
        }
    
    def plot_regimes(self, classes, portfolio=None):
        """
        Plot price, KAMA, and regimes using Plotly.
        
        Parameters:
        -----------
        classes : pandas.DataFrame
            DataFrame with regime classifications
        portfolio : pandas.DataFrame, optional
            DataFrame with strategy performance (default: None)
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('BTC Price, KAMA, and Market Regimes', 
                                          'Strategy Performance vs Buy & Hold' if portfolio is not None else ''))
        
        # Convert index to datetime if needed
        if not isinstance(classes.index, pd.DatetimeIndex):
            classes.index = pd.to_datetime(classes.index)
        
        # Calculate BTC price from log values
        btc_price = np.exp(classes['Log_BTC'])
        
        # Define regime colors
        regime_colors = {
            'Bullish_Low_Var': 'green',
            'Bullish_High_Var': 'yellow',
            'Bearish_Low_Var': 'orange',
            'Bearish_High_Var': 'red'
        }
        
        # Add BTC price trace
        fig.add_trace(
            go.Scatter(
                x=btc_price.index, 
                y=btc_price, 
                name='BTC Price', 
                line=dict(color='blue', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Add KAMA trace
        fig.add_trace(
            go.Scatter(
                x=classes.index, 
                y=np.exp(classes['KAMA']), 
                name='KAMA', 
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add markers for each regime that follow the BTC price
        for label, color in regime_colors.items():
            # Filter data points for this regime
            regime_data = classes[classes['label'] == label]
            
            if not regime_data.empty:
                # Get corresponding prices
                regime_prices = np.exp(regime_data['Log_BTC'])
                
                # Add scatter plot with markers
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_prices,
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=4,
                            opacity=0.7,
                            line=dict(width=1, color='black')
                        ),
                        name=label.replace('_', ' '),
                        hovertemplate='%{x}<br>Price: %{y:$,.2f}<br>Regime: ' + label.replace('_', ' ')
                    ),
                    row=1, col=1
                )
        
        # Add strategy performance if provided
        if portfolio is not None:
            # Add strategy trace
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index, 
                    y=portfolio['strategy'], 
                    name='Strategy', 
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            # Add buy & hold trace
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index, 
                    y=portfolio['btc_price'] / portfolio['btc_price'].iloc[0], 
                    name='Buy & Hold', 
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis_title="BTC Price (USD)",
            yaxis2_title="Normalized Value" if portfolio is not None else "",
        )
        
        # Update x-axis to show date format
        fig.update_xaxes(
            tickformat="%Y-%m",
            tickangle=45,
            tickmode="auto"
        )
        
        # Show the plot
        fig.show()
        
    def save_model(self, filename=None):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filename : str, optional
            File path to save the model (default: None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"markov_kama_model_{timestamp}.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save model
        with open(filename, 'wb') as f:
            pickle.dump({
                'markov': self.markov if hasattr(self, 'markov') else None,
                'kama_values': self.kama_values if hasattr(self, 'kama_values') else None,
                'filtr': self.filtr if hasattr(self, 'filtr') else None,
                'periods_low_high': self.periods_low_high if hasattr(self, 'periods_low_high') else None,
                'aligned_classes': self.aligned_classes if hasattr(self, 'aligned_classes') else None,
                'msreg_results': self.msreg_results if hasattr(self, 'msreg_results') else None
            }, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, data_path, model_path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        data_path : str
            Path to the data file
        model_path : str
            Path to the saved model file
            
        Returns:
        --------
        MarkovKAMA
            Loaded model
        """
        # Create instance
        instance = cls(data_path)
        
        # Load saved model
        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)
        
        # Restore model attributes
        for key, value in saved_model.items():
            setattr(instance, key, value)
        
        print(f"Model loaded from {model_path}")
        
        return instance
    
    def plot_backtest(self, portfolio, classes):
        """
        Create a comprehensive backtest visualization similar to the provided example.
        
        Parameters:
        -----------
        portfolio : pandas.DataFrame
            DataFrame with strategy performance data
        classes : pandas.DataFrame
            DataFrame with regime classifications
        """
        # Create subplots
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.25, 0.15, 0.15, 0.1],
            subplot_titles=(
                'Portfolio Performance',
                'Price with Regimes',
                'Trading State',
                'Predicted State Confidence',
                'Cumulative Trading Costs'
            )
        )
        
        # 1. Portfolio Performance
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['strategy_normalized'],
                name='Strategy',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['buy_hold_normalized'],
                name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=[1] * len(portfolio),
                name='Initial Capital',
                line=dict(color='black', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Price with Regimes
        # Convert to numeric for coloring
        regime_map = {
            'Bullish_Low_Var': 2, 
            'Bullish_High_Var': 1,
            'Bearish_Low_Var': 0, 
            'Bearish_High_Var': -1
        }
        
        # Add current position marker
        current_position = "Flat" if portfolio['trading_state'].iloc[-1] == 0 else "Long"
        current_date = portfolio.index[-1].strftime('%Y-%m-%d')
        
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text=f"Current Position: {current_position} (as of {current_date})",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Add BTC price trace
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['btc_price'],
                name='BTC Price',
                line=dict(color='gray', width=1.5)
            ),
            row=2, col=1
        )
        
        # Add colored markers for regimes
        regime_colors = {
            'Bullish_Low_Var': 'green',
            'Bullish_High_Var': 'yellow',
            'Bearish_Low_Var': 'orange',
            'Bearish_High_Var': 'red'
        }
        
        for regime, color in regime_colors.items():
            regime_data = portfolio[portfolio['regime'] == regime]
            if not regime_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data['btc_price'],
                        mode='markers',
                        marker=dict(color=color, size=5),
                        name=regime.replace('_', ' '),
                        hoverinfo='text',
                        hovertext=[f"Date: {d}<br>Price: ${p:.2f}<br>Regime: {r}" 
                                 for d, p, r in zip(regime_data.index.strftime('%Y-%m-%d'), 
                                                   regime_data['btc_price'], 
                                                   regime_data['regime'])]
                    ),
                    row=2, col=1
                )
        
        # 3. Trading State
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['trading_state'],
                name='Trading State',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        # Add trade markers
        trades = portfolio[portfolio['trades'] == 1]
        if not trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=trades.index,
                    y=trades['trading_state'],
                    mode='markers',
                    marker=dict(
                        color='black',
                        symbol='circle',
                        size=8,
                        line=dict(color='white', width=1)
                    ),
                    name='Trades'
                ),
                row=3, col=1
            )
        
        # 4. Predicted State Confidence (from Markov model)
        if hasattr(self, 'markov'):
            # Get the relevant dates from portfolio
            markov_subset = self.markov.reindex(portfolio.index, method='ffill')
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=markov_subset['High_var'] if 'High_var' in markov_subset.columns else [0]*len(portfolio),
                    name='High Volatility Confidence',
                    line=dict(color='purple', width=2)
                ),
                row=4, col=1
            )
            
            # Add threshold line
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=[0.5] * len(portfolio),
                    name='Threshold',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=4, col=1
            )
        
        # 5. Cumulative Trading Costs
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['cumulative_costs'],
                name='Cumulative Costs',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=5, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text='Strategy Backtest Results',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=120, b=50),
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="BTC Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="State", tickvals=[0, 1], ticktext=["Flat", "Long"], row=3, col=1)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=4, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=5, col=1)
        
        # Update x-axis to show date format
        fig.update_xaxes(
            tickformat="%b %Y",
            tickangle=45,
            tickmode="auto",
            row=5, col=1
        )
        
        # Show the figure
        fig.show()
        
        # Return the figure for further customization if needed
        return fig
    
    def plot_test_results(self, portfolio, classes, figsize=(1000, 800)):
        """
        Plot test results with a clean modern visualization.
        
        Parameters:
        -----------
        portfolio : pandas.DataFrame
            DataFrame with strategy performance data
        classes : pandas.DataFrame
            DataFrame with regime classifications
        figsize : tuple
            Figure size (width, height) (default: (1000, 800))
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The generated figure
        """
        # Filter for test data only
        if 'dataset' in self.markov.columns:
            test_indices = self.markov[self.markov['dataset'] == 2].index
            
            # Filter portfolio and classes for test data only
            test_portfolio = portfolio[portfolio.index.isin(test_indices)]
            test_classes = classes[classes.index.isin(test_indices)]
        else:
            print("Warning: Dataset markers not found. Using all data.")
            test_portfolio = portfolio
            test_classes = classes
        
        # Initialize both portfolios at 1000 at the start of test data
        initial_value = 1000
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=(
                "Portfolio Performance", 
                "BTC Price with Regime Classification", 
                "Cumulative Trading Costs"
            )
        )
        
        # Calculate metrics for display
        metrics = self.calculate_performance_metrics(test_portfolio, test_only=True)
        
        # Calculate buy & hold stats
        btc_metrics = {
            "Sharpe": metrics["BTC Sharpe Ratio"],
            "Sortino": metrics["BTC Sortino Ratio"],
            "Max DD": metrics["BTC Max Drawdown"],
            "Final Value": metrics["Final Buy & Hold Value"] * initial_value
        }
        
        # Format strategy metrics
        strategy_metrics = {
            "Sharpe": metrics["Strategy Sharpe Ratio"],
            "Sortino": metrics["Strategy Sortino Ratio"],
            "Max DD": metrics["Strategy Max Drawdown"],
            "Trades": metrics["Number of Trades"],
            "Costs": metrics["Total Trading Costs"],
            "Accuracy": metrics["Regime Accuracy"],
            "Final Value": metrics["Final Strategy Value"] * initial_value
        }
        
        # 1. Portfolio Performance - only strategy and buy & hold starting at 1000
        # Add strategy portfolio value
        strategy_values = test_portfolio['strategy_normalized'] / test_portfolio['strategy_normalized'].iloc[0] * initial_value
        
        fig.add_trace(
            go.Scatter(
                x=test_portfolio.index,
                y=strategy_values,
                name="Strategy",
                line=dict(color='#003366', width=2),  # Dark blue
                showlegend=False  # Hide from legend
            ),
            row=1, col=1
        )
        
        # Add buy & hold value
        buyhold_values = test_portfolio['buy_hold_normalized'] / test_portfolio['buy_hold_normalized'].iloc[0] * initial_value
        
        fig.add_trace(
            go.Scatter(
                x=test_portfolio.index,
                y=buyhold_values,
                name="Buy & Hold",
                line=dict(color='#e63946', width=2),  # Red
                showlegend=False  # Hide from legend
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_trace(
            go.Scatter(
                x=test_portfolio.index,
                y=[initial_value] * len(test_portfolio),
                name="Initial Capital",
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False  # Hide from legend
            ),
            row=1, col=1
        )
        
        # 2. BTC Price with Regime Classification
        # Add BTC price line
        fig.add_trace(
            go.Scatter(
                x=test_portfolio.index,
                y=test_portfolio['btc_price'],
                name="BTC Price",
                line=dict(color='gray', width=1),
                showlegend=False  # Hide from legend
            ),
            row=2, col=1
        )
        
        # Add colored dots for regimes - with darker yellow
        regime_colors = {
            'Bullish_Low_Var': '#4daf4a',    # Green
            'Bullish_High_Var': '#d4ac0d',   # Darker yellow
            'Bearish_Low_Var': '#ff7f00',    # Orange
            'Bearish_High_Var': '#e41a1c'    # Red
        }
        
        for regime, color in regime_colors.items():
            regime_data = test_portfolio[test_portfolio['regime'] == regime]
            if not regime_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data['btc_price'],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=8,
                            line=dict(width=1, color='white')
                        ),
                        name=regime.replace('_', ' '),
                        hovertemplate='%{x}<br>Price: $%{y:,.2f}<br>Regime: ' + regime.replace('_', ' '),
                        showlegend=False  # Hide from legend
                    ),
                    row=2, col=1
                )
        
        # 3. Cumulative Trading Costs
        fig.add_trace(
            go.Scatter(
                x=test_portfolio.index,
                y=test_portfolio['cumulative_costs'],
                name="Trading Costs",
                line=dict(color='#d62828', width=2),  # Red
                fill='tozeroy',
                fillcolor='rgba(214, 40, 40, 0.1)',
                showlegend=False  # Hide from legend
            ),
            row=3, col=1
        )
        
        # Add trade markers
        trades = test_portfolio[test_portfolio['trades'] == 1]
        if not trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=trades.index,
                    y=trades['cumulative_costs'],
                    mode='markers',
                    marker=dict(
                        color='black',
                        symbol='circle',
                        size=8,
                        line=dict(color='white', width=1)
                    ),
                    name='Trade Points',
                    hovertemplate='%{x}<br>Cumulative Cost: $%{y:.2f}',
                    showlegend=False  # Hide from legend
                ),
                row=3, col=1
            )
        
        # Update layout for a clean modern look
        fig.update_layout(
            height=figsize[1],
            width=figsize[0],
            template='plotly_white',
            title=dict(
                text="Strategy Backtest Results",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,  # Disable legend completely
            margin=dict(l=60, r=60, t=80, b=150)  # Significantly increased bottom margin
        )
        
        # Update axes
        fig.update_yaxes(title_text="Portfolio Value ($)", gridcolor='lightgray', row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", gridcolor='lightgray', row=2, col=1)
        fig.update_yaxes(title_text="Cost ($)", gridcolor='lightgray', row=3, col=1)
        
        # Adjust the bottom x-axis to ensure dates are visible
        fig.update_xaxes(
            gridcolor='lightgray',
            tickformat="%b %Y",
            tickangle=45,
            tickmode="auto",
            showgrid=True,
            row=3, col=1
        )
        
        # Format metrics text
        strategy_text = (
            f"<b>Strategy:</b> Sharpe: {strategy_metrics['Sharpe']:.2f} | "
            f"Sortino: {strategy_metrics['Sortino']:.2f} | "
            f"Max DD: {strategy_metrics['Max DD']:.2%} | "
            f"Trades: {strategy_metrics['Trades']} | "
            f"Costs: ${strategy_metrics['Costs']:.2f} | "
            f"Accuracy: {strategy_metrics['Accuracy']:.2%}"
        )
        
        buy_hold_text = (
            f"<b>Buy & Hold:</b> Sharpe: {btc_metrics['Sharpe']:.2f} | "
            f"Sortino: {btc_metrics['Sortino']:.2f} | "
            f"Max DD: {btc_metrics['Max DD']:.2%}"
        )
        
        # Position the table much lower to avoid collision with date labels
        fig.add_annotation(
            x=0.5, y=-0.25,  # Significantly lower position to avoid date labels
            xref="paper", yref="paper",
            text=f"{strategy_text}<br>{buy_hold_text}",
            showarrow=False,
            font=dict(size=12),
            align="center",
            bgcolor="rgba(248, 249, 250, 0.9)",
            bordercolor="#343a40",
            borderwidth=1,
            borderpad=8
        )
        
        # Show the figure
        fig.show()
        
        return fig
