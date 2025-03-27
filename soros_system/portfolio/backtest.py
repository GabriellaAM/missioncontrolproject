import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from ..analysis.metrics import MetricsCalculator

class PortfolioBacktester:
    """
    Class for backtesting portfolio performance.
    """
    def __init__(self, portfolio_manager, data_loader):
        """
        Initialize the PortfolioBacktester.
        
        Args:
            portfolio_manager: PortfolioManager instance
            data_loader: DataLoader instance
        """
        self.logger = logging.getLogger(__name__)
        self.portfolio_manager = portfolio_manager
        self.data_loader = data_loader
        self.metrics_calculator = MetricsCalculator()
    
    def backtest_portfolio(self, portfolio_name, start_date, end_date, initial_capital=10000, 
                          alt_cost=0.005, btc_cost=0.001, signal_threshold=75):
        """
        Backtest a portfolio over a specific time period.
        
        Args:
            portfolio_name (str): Name of the portfolio to backtest
            start_date (str or datetime): Start date for backtesting
            end_date (str or datetime): End date for backtesting
            initial_capital (float, optional): Initial capital for backtesting
            alt_cost (float, optional): Transaction cost for altcoins
            btc_cost (float, optional): Transaction cost for Bitcoin
            signal_threshold (int, optional): Threshold for signal strength
            
        Returns:
            dict: Backtest results
        """
        # Get portfolio details
        portfolio = self.portfolio_manager.get_portfolio_details(portfolio_name)
        if not portfolio:
            self.logger.warning(f"Portfolio '{portfolio_name}' not found.")
            return None
        
        # Convert dates to pandas datetime
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Get criteria from portfolio
        criteria = portfolio.get('criteria', {})
        
        # Process BTC first
        btc_data = self.data_loader.load_asset_data('bitcoin')
        if btc_data.empty:
            self.logger.error("Failed to load Bitcoin data.")
            return None
        
        # Filter data by date range
        btc_data = btc_data[(btc_data['date'] >= start_date) & (btc_data['date'] <= end_date)]
        if btc_data.empty:
            self.logger.error("No Bitcoin data available in the specified date range.")
            return None
        
        # Initialize results
        results = {
            'portfolio_name': portfolio_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'dates': btc_data['date'].tolist(),
            'btc_prices': btc_data['close'].tolist(),
            'portfolio_values': [],
            'holdings': [],
            'returns': [],
            'trades': [],
            'signals': [],  # Add signals tracking
            'asset_performance': {},
            'metrics': {},
            'asset_trade_history': {}  # Add detailed trade history
        }
        
        # Determine assets to include
        assets = self._get_portfolio_assets(portfolio, criteria)
        
        if not assets:
            self.logger.warning(f"No assets found for portfolio '{portfolio_name}'.")
            return results
        
        # Set initial state
        capital = initial_capital
        holdings = {asset_id: 0 for asset_id in assets}
        holdings['cash'] = capital
        
        # Initialize asset data cache
        asset_data_cache = {}
        
        # Initialize asset trade history
        trade_id_counter = {asset: 0 for asset in assets}
        open_trades = {asset: None for asset in assets}
        
        # Process each date
        for i, row in btc_data.iterrows():
            date = row['date']
            
            # Get BTC trend and volatility state if needed
            btc_trend = self._get_btc_trend(btc_data, date, criteria)
            vol_state = self._get_volatility(date, criteria)
            
            # Track signals for all assets on this date
            date_signals = []
            
            # Check if we should apply BTC gating
            btc_gate_active = False
            if criteria.get('btc_trend_gating') is not None and btc_trend < criteria.get('btc_trend_gating'):
                btc_gate_active = True
                
                # Sell all holdings if BTC trend is below threshold
                for asset_id in assets:
                    if holdings.get(asset_id, 0) > 0:
                        # Get asset price
                        asset_price = self._get_asset_price(asset_id, date, asset_data_cache)
                        if asset_price:
                            # Sell asset
                            sale_value = holdings[asset_id] * asset_price * (1 - alt_cost)
                            holdings['cash'] += sale_value
                            
                            # Record trade
                            results['trades'].append({
                                'date': date,
                                'asset': asset_id,
                                'action': 'SELL',
                                'amount': holdings[asset_id],
                                'price': asset_price,
                                'value': sale_value,
                                'reason': 'BTC_GATE'
                            })
                            
                            # Record trade in asset trade history
                            if open_trades[asset_id]:
                                trade = open_trades[asset_id].copy()
                                trade['exit_date'] = date
                                trade['exit_price'] = asset_price
                                trade['holding_days'] = (pd.Timestamp(date) - pd.Timestamp(trade['entry_date'])).days
                                trade['price_return'] = f"{((asset_price / trade['entry_price']) - 1) * 100:.2f}%"
                                trade['exit_value'] = holdings[asset_id] * asset_price
                                trade['exit_cost'] = holdings[asset_id] * asset_price * alt_cost
                                trade['trade_return'] = f"{((trade['exit_value'] - trade['exit_cost']) / (trade['entry_value'] + trade['entry_cost']) - 1) * 100:.2f}%"
                                trade['is_open'] = False
                                
                                if asset_id not in results['asset_trade_history']:
                                    results['asset_trade_history'][asset_id] = []
                                
                                results['asset_trade_history'][asset_id].append(trade)
                                open_trades[asset_id] = None
                            
                            holdings[asset_id] = 0
                
                # Calculate portfolio value
                portfolio_value = holdings['cash']
                results['portfolio_values'].append(portfolio_value)
                results['holdings'].append(holdings.copy())
                
                # Calculate return
                if i > 0:
                    daily_return = (portfolio_value / results['portfolio_values'][-2]) - 1
                    results['returns'].append(daily_return)
                else:
                    results['returns'].append(0)
            
            # Apply asset selection logic
            asset_signals = {}
            for asset_id in assets:
                # Get asset data
                asset_data = self._get_asset_data(asset_id, asset_data_cache)
                if asset_data.empty:
                    continue
                
                # Filter data by date
                asset_date_data = asset_data[asset_data['date'] <= date].tail(30)  # Use last 30 days
                if asset_date_data.empty:
                    continue
                
                # Calculate signals
                usd_trend_signal, btc_trend_signal, rsi_usd_signal, rsi_btc_signal, \
                volatility_signal, ssr_signal, ssr_gate, btc_rsi_signal, combined_signal = \
                    self._calculate_detailed_signals(asset_id, asset_date_data, criteria, btc_trend, vol_state)
                
                # Store all signal components
                signals_dict = {
                    'date': date,
                    'asset': asset_id,
                    'usd_trend_signal': usd_trend_signal,
                    'btc_trend_signal': btc_trend_signal,
                    'rsi_usd_signal': rsi_usd_signal,
                    'rsi_btc_signal': rsi_btc_signal,
                    'volatility_signal': volatility_signal,
                    'ssr_signal': ssr_signal,
                    'ssr_gate': ssr_gate,
                    'btc_gate': btc_gate_active,
                    'btc_rsi_gate': criteria.get('use_btc_rsi_signal'),
                    'btc_rsi_signal': btc_rsi_signal,
                    'followed_portfolio_signal': None,  # Add support for following another portfolio
                    'combined_signal': combined_signal,
                    'final_decision': 1 if not btc_gate_active and combined_signal >= signal_threshold else 0
                }
                
                # Store signal
                date_signals.append(signals_dict)
                
                # Store asset signal for allocating capital
                asset_signals[asset_id] = combined_signal
            
            # Add all signals for this date
            results['signals'].extend(date_signals)
            
            # If BTC gate is active, skip asset allocation
            if btc_gate_active:
                continue
            
            # Sort assets by signal strength
            sorted_assets = sorted(
                [(asset_id, signal) for asset_id, signal in asset_signals.items() if signal >= signal_threshold],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Determine trade actions
            target_positions = {}
            if sorted_assets:
                # Allocate capital among top assets
                num_assets = min(len(sorted_assets), criteria.get('max_assets', 5))
                allocation_per_asset = 1.0 / num_assets if num_assets > 0 else 0
                
                for i in range(num_assets):
                    asset_id = sorted_assets[i][0]
                    target_positions[asset_id] = allocation_per_asset
            
            # Calculate portfolio value
            portfolio_value = holdings['cash']
            for asset, amount in holdings.items():
                if asset != 'cash' and amount > 0:
                    asset_price = self._get_asset_price(asset, date, asset_data_cache)
                    if asset_price:
                        portfolio_value += amount * asset_price
            
            # Execute trades
            for asset_id in assets:
                target_allocation = target_positions.get(asset_id, 0)
                current_allocation = 0
                
                # Calculate current allocation
                if portfolio_value > 0 and asset_id in holdings and holdings[asset_id] > 0:
                    asset_price = self._get_asset_price(asset_id, date, asset_data_cache)
                    if asset_price:
                        current_allocation = (holdings[asset_id] * asset_price) / portfolio_value
                
                # Determine if we need to trade
                if abs(current_allocation - target_allocation) > 0.01:  # 1% threshold
                    if target_allocation > current_allocation:
                        # Buy more
                        amount_to_allocate = (target_allocation - current_allocation) * portfolio_value
                        if amount_to_allocate > 0 and holdings['cash'] >= amount_to_allocate:
                            asset_price = self._get_asset_price(asset_id, date, asset_data_cache)
                            if asset_price:
                                # Calculate amount to buy
                                buy_amount = (amount_to_allocate * (1 - alt_cost)) / asset_price
                                
                                # Execute buy
                                holdings[asset_id] = holdings.get(asset_id, 0) + buy_amount
                                holdings['cash'] -= amount_to_allocate
                                
                                # Record trade
                                trade_info = {
                                    'date': date,
                                    'asset': asset_id,
                                    'action': 'BUY',
                                    'amount': buy_amount,
                                    'price': asset_price,
                                    'value': amount_to_allocate,
                                    'reason': 'SIGNAL'
                                }
                                results['trades'].append(trade_info)
                                
                                # Record trade in asset trade history
                                if asset_id not in results['asset_trade_history']:
                                    results['asset_trade_history'][asset_id] = []
                                
                                # Create a new trade record
                                trade_id_counter[asset_id] += 1
                                trade = {
                                    'trade_id': trade_id_counter[asset_id],
                                    'entry_date': date,
                                    'exit_date': None,
                                    'holding_days': 0,
                                    'entry_price': asset_price,
                                    'exit_price': None,
                                    'price_return': None,
                                    'entry_value': amount_to_allocate * (1 - alt_cost),
                                    'entry_cost': amount_to_allocate * alt_cost,
                                    'exit_value': None,
                                    'exit_cost': None,
                                    'trade_return': None,
                                    'is_open': True
                                }
                                
                                open_trades[asset_id] = trade
                    else:
                        # Sell some or all
                        current_value = holdings[asset_id] * self._get_asset_price(asset_id, date, asset_data_cache)
                        target_value = target_allocation * portfolio_value
                        amount_to_sell = (current_value - target_value) / self._get_asset_price(asset_id, date, asset_data_cache)
                        
                        if amount_to_sell > 0:
                            # Ensure we don't sell more than we have
                            amount_to_sell = min(amount_to_sell, holdings[asset_id])
                            
                            # Execute sell
                            sale_price = self._get_asset_price(asset_id, date, asset_data_cache)
                            sale_value = amount_to_sell * sale_price * (1 - alt_cost)
                            
                            holdings[asset_id] -= amount_to_sell
                            holdings['cash'] += sale_value
                            
                            # Record trade
                            trade_info = {
                                'date': date,
                                'asset': asset_id,
                                'action': 'SELL',
                                'amount': amount_to_sell,
                                'price': sale_price,
                                'value': sale_value,
                                'reason': 'REALLOCATION'
                            }
                            results['trades'].append(trade_info)
                            
                            # Record trade in asset trade history
                            if open_trades[asset_id]:
                                trade = open_trades[asset_id].copy()
                                trade['exit_date'] = date
                                trade['exit_price'] = sale_price
                                trade['holding_days'] = (pd.Timestamp(date) - pd.Timestamp(trade['entry_date'])).days
                                trade['price_return'] = f"{((sale_price / trade['entry_price']) - 1) * 100:.2f}%"
                                trade['exit_value'] = amount_to_sell * sale_price
                                trade['exit_cost'] = amount_to_sell * sale_price * alt_cost
                                trade['trade_return'] = f"{((trade['exit_value'] - trade['exit_cost']) / (trade['entry_value'] + trade['entry_cost']) - 1) * 100:.2f}%"
                                trade['is_open'] = False
                                
                                if asset_id not in results['asset_trade_history']:
                                    results['asset_trade_history'][asset_id] = []
                                
                                results['asset_trade_history'][asset_id].append(trade)
                                
                                # If we still have some position, create a new open trade for the remainder
                                if holdings[asset_id] > 0:
                                    trade_id_counter[asset_id] += 1
                                    open_trades[asset_id] = {
                                        'trade_id': trade_id_counter[asset_id],
                                        'entry_date': date,
                                        'exit_date': None,
                                        'holding_days': 0,
                                        'entry_price': sale_price,  # Use current price as entry
                                        'exit_price': None,
                                        'price_return': None,
                                        'entry_value': holdings[asset_id] * sale_price,
                                        'entry_cost': 0,  # No cost for this portion as it's a continuation
                                        'exit_value': None,
                                        'exit_cost': None,
                                        'trade_return': None,
                                        'is_open': True
                                    }
                                else:
                                    open_trades[asset_id] = None
            
            # Calculate portfolio value
            portfolio_value = holdings['cash']
            for asset, amount in holdings.items():
                if asset != 'cash' and amount > 0:
                    asset_price = self._get_asset_price(asset, date, asset_data_cache)
                    if asset_price:
                        portfolio_value += amount * asset_price
            
            # Record results
            results['portfolio_values'].append(portfolio_value)
            results['holdings'].append(holdings.copy())
            
            # Calculate return
            if i > 0:
                daily_return = (portfolio_value / results['portfolio_values'][-2]) - 1
                results['returns'].append(daily_return)
            else:
                results['returns'].append(0)
        
        # Close any open trades at the end of the backtest
        for asset_id, trade in open_trades.items():
            if trade:
                final_date = end_date
                asset_price = self._get_asset_price(asset_id, final_date, asset_data_cache)
                
                if asset_price:
                    trade['exit_date'] = final_date
                    trade['exit_price'] = asset_price
                    trade['holding_days'] = (pd.Timestamp(final_date) - pd.Timestamp(trade['entry_date'])).days
                    trade['price_return'] = f"{((asset_price / trade['entry_price']) - 1) * 100:.2f}%"
                    
                    position_size = holdings.get(asset_id, 0)
                    trade['exit_value'] = position_size * asset_price
                    trade['exit_cost'] = position_size * asset_price * alt_cost
                    
                    if trade['entry_value'] > 0:
                        trade['trade_return'] = f"{((trade['exit_value'] - trade['exit_cost']) / (trade['entry_value'] + trade['entry_cost']) - 1) * 100:.2f}%"
                    else:
                        trade['trade_return'] = "0.00%"
                    
                    trade['is_open'] = False
                    
                    if asset_id not in results['asset_trade_history']:
                        results['asset_trade_history'][asset_id] = []
                    
                    results['asset_trade_history'][asset_id].append(trade)
        
        # Calculate performance metrics
        results['metrics'] = self._calculate_performance_metrics(results)
        
        # Calculate asset performance
        results['asset_performance'] = self._calculate_asset_performance(results)
        
        # Convert results to pandas DataFrames
        results['results_df'] = pd.DataFrame({
            'Date': results['dates'],
            'Portfolio_Value': results['portfolio_values'],
            'Returns': results['returns']
        })
        
        return results
    
    def _get_portfolio_assets(self, portfolio, criteria):
        """
        Get the list of assets for a portfolio.
        
        Args:
            portfolio (dict): Portfolio details
            criteria (dict): Portfolio criteria
            
        Returns:
            list: List of asset IDs
        """
        # Check for explicitly defined assets
        if 'assets' in portfolio and portfolio['assets']:
            return portfolio['assets']
        
        # If BTC only, return just Bitcoin
        if criteria.get('btc_only', False):
            return ['bitcoin']
        
        # Otherwise, return all assets available
        return [asset_id for asset_id in self.data_loader.asset_data_cache.keys()]
    
    def _get_asset_data(self, asset_id, cache):
        """
        Get data for a specific asset, using cache if available.
        
        Args:
            asset_id (str): Asset ID
            cache (dict): Cache for asset data
            
        Returns:
            pd.DataFrame: Asset data
        """
        if asset_id in cache:
            return cache[asset_id]
        
        asset_data = self.data_loader.load_asset_data(asset_id)
        cache[asset_id] = asset_data
        
        return asset_data
    
    def _get_asset_price(self, asset_id, date, cache):
        """
        Get price for a specific asset on a specific date.
        
        Args:
            asset_id (str): Asset ID
            date (datetime): Date to get price for
            cache (dict): Cache for asset data
            
        Returns:
            float: Asset price, or None if not available
        """
        asset_data = self._get_asset_data(asset_id, cache)
        if asset_data.empty:
            return None
        
        # Find closest date
        closest_data = asset_data[asset_data['date'] <= date].tail(1)
        if closest_data.empty:
            return None
        
        return closest_data['close'].iloc[0]
    
    def _get_btc_trend(self, btc_data, date, criteria):
        """
        Get Bitcoin trend for a specific date.
        
        Args:
            btc_data (pd.DataFrame): Bitcoin data
            date (datetime): Date to get trend for
            criteria (dict): Portfolio criteria
            
        Returns:
            int: Bitcoin trend (-2 to 2)
        """
        # Find closest date
        closest_data = btc_data[btc_data['date'] <= date].tail(1)
        if closest_data.empty:
            return 0
        
        # Get trend column
        trend_col = 'Overall_Trend_USD'
        if trend_col not in closest_data.columns:
            return 0
        
        return closest_data[trend_col].iloc[0]
    
    def _get_volatility(self, date, criteria):
        """
        Get volatility state for a specific date.
        
        Args:
            date (datetime): Date to get volatility for
            criteria (dict): Portfolio criteria
            
        Returns:
            int: Volatility state (0 for low, 1 for high)
        """
        # Check if we have a Markov model
        if hasattr(self, 'markov_analyzer') and self.markov_analyzer:
            return self.markov_analyzer.get_volatility_state(date)
        
        return 0  # Default to low volatility
    
    def _calculate_detailed_signals(self, asset_id, asset_data, criteria, btc_trend, vol_state):
        """
        Calculate detailed signals for an asset.
        
        Args:
            asset_id (str): Asset ID
            asset_data (pd.DataFrame): Asset data
            criteria (dict): Portfolio criteria
            btc_trend (int): Current BTC trend
            vol_state (int): Current volatility state
            
        Returns:
            tuple: (usd_trend_signal, btc_trend_signal, rsi_usd_signal, rsi_btc_signal, volatility_signal, ssr_signal, ssr_gate, btc_rsi_signal, combined_signal)
        """
        # Initialize signals
        usd_trend_signal = 0
        btc_trend_signal = 0
        rsi_usd_signal = 0
        rsi_btc_signal = 0
        volatility_signal = 0
        ssr_signal = None
        ssr_gate = None
        btc_rsi_signal = None
        
        # Get latest data
        latest_data = asset_data.iloc[-1] if not asset_data.empty else None
        
        if latest_data is None:
            return usd_trend_signal, btc_trend_signal, rsi_usd_signal, rsi_btc_signal, volatility_signal, ssr_signal, ssr_gate, btc_rsi_signal, 0
        
        # USD trend signal
        if 'Overall_Trend_USD' in latest_data:
            # Process trend conditions format
            usd_conditions = criteria.get('usd_conditions', [])
            
            # If criteria defined by trend terms
            if isinstance(usd_conditions, dict):
                # Check each trend term separately
                for term, conditions in usd_conditions.items():
                    col_name = f"Trend_{term}_USD"
                    if col_name in latest_data and latest_data[col_name] in conditions:
                        usd_trend_signal = 100  # Match found
                        break
            # If general trend list
            elif latest_data['Overall_Trend_USD'] in usd_conditions:
                usd_trend_signal = 100
        
        # BTC trend signal
        if 'Overall_Trend_BTC' in latest_data:
            # Process trend conditions format
            btc_conditions = criteria.get('btc_conditions', [])
            
            # If criteria defined by trend terms
            if isinstance(btc_conditions, dict):
                # Check each trend term separately
                for term, conditions in btc_conditions.items():
                    col_name = f"Trend_{term}_BTC"
                    if col_name in latest_data and latest_data[col_name] in conditions:
                        btc_trend_signal = 100  # Match found
                        break
            # If general trend list
            elif latest_data['Overall_Trend_BTC'] in btc_conditions:
                btc_trend_signal = 100
        
        # RSI signals
        if criteria.get('rsi_conditions_usd') and 'rsi_14_close' in latest_data:
            rsi = latest_data['rsi_14_close']
            if rsi < 30:  # Oversold
                rsi_usd_signal = 100
            elif rsi > 70:  # Overbought
                rsi_usd_signal = -100
            else:
                rsi_usd_signal = 0
        
        if criteria.get('rsi_conditions_btc') and f'rsi_14_{asset_id}_btc' in latest_data:
            rsi = latest_data[f'rsi_14_{asset_id}_btc']
            if rsi < 30:  # Oversold
                rsi_btc_signal = 100
            elif rsi > 70:  # Overbought
                rsi_btc_signal = -100
            else:
                rsi_btc_signal = 0
        
        # Volatility signal
        if criteria.get('use_volatility_filter') and vol_state is not None:
            # Determine if we want high or low volatility
            prefer_high = criteria.get('volatility_weight', 1.0) > 0
            
            if (prefer_high and vol_state == 1) or (not prefer_high and vol_state == 0):
                volatility_signal = 100 * abs(criteria.get('volatility_weight', 1.0))
            else:
                volatility_signal = -100 * abs(criteria.get('volatility_weight', 1.0))
        
        # BTC RSI signal
        if criteria.get('use_btc_rsi_signal'):
            btc_rsi_signal = rsi_usd_signal  # Reuse USD RSI signal for simplicity
        
        # Calculate combined signal
        weights = {
            'usd_trend': 1.0,
            'btc_trend': 0.8,
            'rsi_usd': 0.5,
            'rsi_btc': 0.3,
            'volatility': criteria.get('volatility_weight', 1.0),
            'btc_rsi': 0.5 if criteria.get('use_btc_rsi_signal') else 0.0
        }
        
        signal_values = {
            'usd_trend': usd_trend_signal,
            'btc_trend': btc_trend_signal,
            'rsi_usd': rsi_usd_signal,
            'rsi_btc': rsi_btc_signal,
            'volatility': volatility_signal,
            'btc_rsi': btc_rsi_signal if btc_rsi_signal is not None else 0
        }
        
        # Calculate weighted signal
        total_weight = sum(weight for name, weight in weights.items() if signal_values[name] != 0)
        if total_weight > 0:
            combined_signal = sum(weights[name] * signal_values[name] for name in weights) / total_weight
        else:
            combined_signal = 0
        
        return usd_trend_signal, btc_trend_signal, rsi_usd_signal, rsi_btc_signal, volatility_signal, ssr_signal, ssr_gate, btc_rsi_signal, combined_signal
    
    def _calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for the portfolio backtest.
        
        Args:
            results (dict): Backtest results
            
        Returns:
            dict: Performance metrics
        """
        if not results or 'portfolio_values' not in results or len(results['portfolio_values']) < 2:
            return {}
        
        # Calculate returns
        initial_value = results['initial_capital']
        final_value = results['portfolio_values'][-1]
        total_return_pct = ((final_value / initial_value) - 1) * 100
        
        # Calculate daily returns
        returns = np.array(results['returns'])
        
        # Calculate volatility
        volatility = np.std(returns) * 100
        annualized_volatility = volatility * np.sqrt(365)  # Annualize assuming daily returns
        
        # Calculate maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        mean_return = np.mean(returns)
        sharpe_ratio = (mean_return / np.std(returns)) * np.sqrt(365) if np.std(returns) > 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        sortino_ratio = (mean_return / downside_deviation) * np.sqrt(365) if downside_deviation > 0 else 0
        
        # Calculate win rate
        trades = results.get('trades', [])
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        num_trades = len(buy_trades)
        trade_profits = []
        
        if 'asset_trade_history' in results:
            # Get trade returns from asset_trade_history
            total_trades = 0
            winning_trades = 0
            
            for asset_id, trade_list in results['asset_trade_history'].items():
                for trade in trade_list:
                    if 'is_open' in trade and not trade['is_open']:
                        total_trades += 1
                        
                        # Parse trade return percentage
                        if 'trade_return' in trade:
                            try:
                                return_pct = float(trade['trade_return'].strip('%'))
                                trade_profits.append(return_pct)
                                
                                if return_pct > 0:
                                    winning_trades += 1
                            except ValueError:
                                pass
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Calculate average trade profit/loss
        avg_profit = np.mean([p for p in trade_profits if p > 0]) if any(p > 0 for p in trade_profits) else 0
        avg_loss = np.mean([p for p in trade_profits if p < 0]) if any(p < 0 for p in trade_profits) else 0
        
        # Calculate average holding period
        if 'asset_trade_history' in results:
            holding_days = []
            
            for asset_id, trade_list in results['asset_trade_history'].items():
                for trade in trade_list:
                    if 'holding_days' in trade and not trade.get('is_open', False):
                        holding_days.append(trade['holding_days'])
            
            avg_holding_days = np.mean(holding_days) if holding_days else 0
        else:
            avg_holding_days = 0
        
        return {
            'total_return': f"{total_return_pct:.2f}%",
            'total_return_pct': total_return_pct,
            'max_drawdown': f"{max_drawdown:.2f}%",
            'max_drawdown_pct': max_drawdown,
            'volatility': f"{volatility:.2f}%",
            'annualized_volatility': f"{annualized_volatility:.2f}%",
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'initial_capital': initial_value,
            'final_value': final_value,
            'win_rate': f"{win_rate:.2f}%",
            'win_rate_pct': win_rate,
            'num_trades': num_trades,
            'avg_profit': f"{avg_profit:.2f}%",
            'avg_loss': f"{avg_loss:.2f}%" if avg_loss < 0 else "0.00%",
            'avg_holding_days': round(avg_holding_days, 1),
            'profit_factor': abs(avg_profit / avg_loss) if avg_loss < 0 else float('inf')
        }
    
    def _calculate_asset_performance(self, results):
        """
        Calculate performance metrics for individual assets.
        
        Args:
            results (dict): Backtest results
            
        Returns:
            dict: Asset performance metrics
        """
        if not results or 'trades' not in results:
            return {}
        
        # Group trades by asset
        trades_by_asset = {}
        for trade in results['trades']:
            asset = trade['asset']
            if asset not in trades_by_asset:
                trades_by_asset[asset] = []
            trades_by_asset[asset].append(trade)
        
        # Initialize performance metrics
        asset_performance = {}
        
        for asset, trades in trades_by_asset.items():
            # Use detailed trade history if available
            if 'asset_trade_history' in results and asset in results['asset_trade_history']:
                trade_history = results['asset_trade_history'][asset]
                
                # Filter to closed trades
                closed_trades = [t for t in trade_history if not t.get('is_open', False)]
                
                if not closed_trades:
                    continue
                
                # Calculate metrics for this asset
                total_return_pct = 0
                max_drawdown = 0
                trade_returns = []
                winning_trades = 0
                holding_days = []
                total_costs = 0
                entry_dates = []
                exit_dates = []
                
                for trade in closed_trades:
                    # Parse trade return percentage
                    if 'trade_return' in trade:
                        try:
                            return_pct = float(trade['trade_return'].strip('%'))
                            trade_returns.append(return_pct)
                            
                            if return_pct > 0:
                                winning_trades += 1
                        except ValueError:
                            pass
                    
                    # Collect holding days
                    if 'holding_days' in trade:
                        holding_days.append(trade['holding_days'])
                    
                    # Collect costs
                    if 'entry_cost' in trade:
                        total_costs += trade['entry_cost']
                    if 'exit_cost' in trade:
                        total_costs += trade['exit_cost']
                    
                    # Collect dates
                    if 'entry_date' in trade:
                        entry_dates.append(pd.Timestamp(trade['entry_date']))
                    if 'exit_date' in trade:
                        exit_dates.append(pd.Timestamp(trade['exit_date']))
                
                # Calculate final metrics
                total_return_pct = sum(trade_returns)
                win_rate = (winning_trades / len(closed_trades)) * 100 if closed_trades else 0
                
                winning_returns = [r for r in trade_returns if r > 0]
                losing_returns = [r for r in trade_returns if r <= 0]
                
                avg_win = np.mean(winning_returns) if winning_returns else 0
                avg_loss = np.mean(losing_returns) if losing_returns else 0
                
                # Calculate Sharpe and Sortino ratios if we have enough data
                if trade_returns:
                    mean_return = np.mean(trade_returns)
                    std_dev = np.std(trade_returns) if len(trade_returns) > 1 else 0.0001
                    
                    # Only negative returns for Sortino
                    downside_returns = [r for r in trade_returns if r < 0]
                    downside_dev = np.std(downside_returns) if len(downside_returns) > 1 else 0.0001
                    
                    sharpe = mean_return / std_dev if std_dev > 0 else 0
                    sortino = mean_return / downside_dev if downside_dev > 0 else 0
                else:
                    sharpe = 0
                    sortino = 0
                
                # Find first and last trade dates
                first_trade = min(entry_dates) if entry_dates else None
                last_trade = max(exit_dates) if exit_dates else None
                
                # Count trading days
                if first_trade and last_trade:
                    trading_days = (last_trade - first_trade).days + 1
                else:
                    trading_days = 0
                
                # Store metrics
                asset_performance[asset] = {
                    'total_return': f"{total_return_pct:.2f}%",
                    'total_return_pct': total_return_pct,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': round(sharpe, 2),
                    'sortino_ratio': round(sortino, 2),
                    'number_of_trades': len(closed_trades),
                    'win_rate': f"{win_rate:.2f}%",
                    'avg_win': f"{avg_win:.2f}%" if avg_win > 0 else "0.00%",
                    'avg_loss': f"{avg_loss:.2f}%" if avg_loss < 0 else "0.00%",
                    'avg_holding_days': round(np.mean(holding_days), 1) if holding_days else 0,
                    'total_costs': f"${total_costs:.2f}",
                    'first_trade': first_trade,
                    'last_trade': last_trade,
                    'trading_days': trading_days,
                    'trade_history': closed_trades
                }
        
        return asset_performance 