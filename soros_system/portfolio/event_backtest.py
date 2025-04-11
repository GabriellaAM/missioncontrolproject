"""
Event-driven portfolio backtester for the Soros System.

This module implements a portfolio backtester that uses event-driven logic
for signal processing, maintaining signal exposure for optimal holding periods.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from tqdm import tqdm

from ..analysis.metrics import MetricsCalculator
from ..signals.signal_event_tracker import SignalEventTracker
from ..signals.signal_registry import get_signal
from ..analysis.forward_returns.signal_evaluator import SignalEvaluator


class EventDrivenBacktester:
    """
    Backtest portfolio strategies using event-driven signal logic.
    
    This class simulates trades based on signal activation events
    and maintains signal exposure for optimal holding periods.
    """
    
    def __init__(self, portfolio_manager=None, data_loader=None, analyzer=None):
        """Initialize the event-driven backtester.
        
        Args:
            portfolio_manager: Manager for portfolios
            data_loader: Loader for asset data
            analyzer: PortfolioAnalyzer instance (new unified interface)
        """
        self.logger = logging.getLogger(__name__)
        self.portfolio_manager = portfolio_manager
        self.data_loader = data_loader
        self.analyzer = analyzer
        self.metrics_calculator = MetricsCalculator()
        self.signal_evaluator = SignalEvaluator()
        self.debug = False  # Add debug flag
        
        # If analyzer is provided, use it for data loading
        if analyzer is not None:
            self.data_loader = analyzer
            self.logger.info("Using PortfolioAnalyzer for data loading")
    
    def backtest_portfolio(
        self,
        portfolio_name: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        initial_capital: float = 10000.0,
        btc_cost: float = 0.001,
        alt_cost: float = 0.005,
        slippage_pct: float = 0.0,
        signal_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Backtest a portfolio using event-driven signal logic.
        
        Args:
            portfolio_name: Name of the portfolio to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for backtesting
            btc_cost: Fee rate for Bitcoin transactions
            alt_cost: Fee rate for altcoin transactions
            slippage_pct: Slippage percentage for trading
            signal_threshold: Threshold for combined signal weight
            
        Returns:
            dict: Dictionary with backtest results
        """
        self.logger.info(
            f"Starting event-driven backtest for portfolio '{portfolio_name}' "
            f"from {start_date} to {end_date}"
        )
        
        # Convert dates to datetime
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return None
        
        # Get signal tables from portfolio manager
        signal_tables = self.portfolio_manager.get_portfolio_signals(portfolio_name)
        if signal_tables is None or not signal_tables:
            self.logger.error(f"No signal tables available for portfolio '{portfolio_name}'")
            return None
        
        # Create result containers
        results = {}
        signals = {}
        trades = {}
        metrics = {}
        basket_selection = pd.DataFrame()
        asset_metadata = {}
        
        # Create signal event tracker
        event_tracker = SignalEventTracker(threshold=signal_threshold)
        
        # First pass: determine optimal holding periods for all signals
        self.logger.info("Analyzing signals to determine optimal holding periods...")
        for asset_id, signal_df in signal_tables.items():
            asset_metadata[asset_id] = {'optimal_periods': {}}
            
            # Get raw asset data for evaluation
            try:
                asset_data = self.data_loader.get_asset_data(asset_id)
                if asset_data is None or asset_data.empty:
                    self.logger.warning(f"No data available for {asset_id}, skipping asset")
                    continue
            except Exception as e:
                self.logger.error(f"Error loading data for {asset_id}: {e}")
                continue
            
            # Process each signal column
            for col in signal_df.columns:
                if not col.endswith('_signal'):
                    continue
                
                # Extract signal name
                signal_name = col[:-7]  # Remove '_signal' suffix
                
                # Get signal instance
                signal_instance = get_signal(signal_name)
                if signal_instance is None:
                    self.logger.warning(f"Signal {signal_name} not found in registry, skipping")
                    continue
                
                # Evaluate signal to find optimal holding period
                try:
                    evaluation = self.signal_evaluator.evaluate_signal(
                        signal_instance, asset_data, asset_id
                    )
                    
                    # Extract optimal holding period
                    optimal_period = evaluation.get('optimal_holding_period', 14)
                    is_effective = evaluation.get('overall_effectiveness', False)
                    
                    # Use default period for ineffective signals
                    if optimal_period <= 0 or not is_effective:
                        optimal_period = 14  # Default fallback for ineffective signals
                        self.logger.debug(f"Signal {signal_name} for {asset_id} is ineffective, using default period of 14 days")
                    
                    # Store in asset metadata
                    asset_metadata[asset_id]['optimal_periods'][col] = {
                        'period': optimal_period,
                        'weight': evaluation.get('weight', 0.0),
                        'is_effective': is_effective
                    }
                    
                    self.logger.info(
                        f"Signal {signal_name} for {asset_id}: "
                        f"optimal holding period = {optimal_period} days, "
                        f"weight = {evaluation.get('weight', 0.0):.4f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating signal {signal_name} for {asset_id}: {e}")
                    # Use default values
                    asset_metadata[asset_id]['optimal_periods'][col] = {
                        'period': 14,
                        'weight': 0.0
                    }
        
        # Get a unified date range across all assets
        all_dates = pd.DatetimeIndex([])
        for asset_id, signal_df in signal_tables.items():
            if isinstance(signal_df.index, pd.DatetimeIndex):
                all_dates = all_dates.union(signal_df.index)
        
        all_dates = all_dates.sort_values()
        all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
        
        if len(all_dates) == 0:
            self.logger.error("No dates in common date range")
            return None
        
        # Second pass: process signals chronologically and track events
        self.logger.info("Processing signals chronologically to detect events...")
        for asset_id, signal_df in tqdm(signal_tables.items(), desc="Processing assets"):
            if asset_id not in asset_metadata:
                self.logger.warning(f"No metadata for {asset_id}, skipping asset")
                continue
                
            # Filter signal data to date range
            signal_df = signal_df.loc[(signal_df.index >= start_date) & (signal_df.index <= end_date)]
            
            if signal_df.empty:
                self.logger.warning(f"No signal data in date range for {asset_id}, skipping asset")
                continue
            
            # Extract signal columns
            signal_cols = [col for col in signal_df.columns if col.endswith('_signal')]
            if not signal_cols:
                self.logger.warning(f"No signal columns found for {asset_id}, skipping asset")
                continue
            
            # Add price data
            try:
                asset_prices = self.data_loader.get_asset_data(
                    asset_id, start_date=start_date, end_date=end_date
                )
                if asset_prices is None or asset_prices.empty:
                    self.logger.warning(f"No price data for {asset_id}, skipping asset")
                    continue
                
                # Merge price data with signal data
                signal_df = signal_df.join(asset_prices[['open', 'close']], how='left')
                
                # Forward fill any missing prices
                signal_df[['open', 'close']] = signal_df[['open', 'close']].ffill()
                
                # Drop rows with NaN prices
                initial_len = len(signal_df)
                signal_df = signal_df.dropna(subset=['open', 'close'])
                
                if len(signal_df) < initial_len:
                    self.logger.warning(
                        f"Dropped {initial_len - len(signal_df)} rows with NaN prices for {asset_id}"
                    )
                
                if signal_df.empty:
                    self.logger.warning(f"No data left after dropping rows with NaN prices for {asset_id}")
                    continue
            except Exception as e:
                self.logger.error(f"Error processing price data for {asset_id}: {e}")
                continue
            
            # Create results dataframe
            asset_results = pd.DataFrame(index=signal_df.index)
            
            # Iterate through dates to detect signal activations
            prev_signals = {}
            for date, row in signal_df.iterrows():
                # Check for signal activations (0 -> 1)
                for signal_col in signal_cols:
                    current_value = row[signal_col]
                    
                    # Get previous value (default to 0)
                    prev_value = prev_signals.get(signal_col, 0)
                    
                    # Check for activation (0 -> 1)
                    if prev_value == 0 and current_value == 1:
                        # Signal activated!
                        self.logger.debug(f"Signal activated: {signal_col} for {asset_id} on {date}")
                        
                        # Get optimal holding period and weight
                        opt_data = asset_metadata[asset_id]['optimal_periods'].get(signal_col, {})
                        holding_period = opt_data.get('period', 14)
                        weight = opt_data.get('weight', 0.0)
                        
                        # Register event
                        event_tracker.register_signal_event(
                            signal_name=signal_col,
                            asset_id=asset_id,
                            activation_date=date,
                            holding_period=holding_period,
                            weight=weight
                        )
                    
                    # Update previous signal
                    prev_signals[signal_col] = current_value
                
                # Calculate final decision for this date
                decision = event_tracker.get_final_decision(asset_id, date)
                
                # Store in results
                asset_results.loc[date, 'event_decision'] = decision
            
            # Store signals for this asset
            signals[asset_id] = signal_df.copy()
            
            # Add event_decision column to signals
            signals[asset_id]['event_decision'] = asset_results['event_decision']
            
            # Simulate trades for this asset with full initial capital
            asset_trades_list = []
            
            # Initial state
            current_cash = initial_capital
            current_holdings_qty = 0.0
            last_entry_price = 0.0
            last_entry_date = None
            trade_id = 0
            
            # Choose fee rate based on asset
            fee_rate = btc_cost if asset_id.lower() == 'btc' else alt_cost
            total_cost_rate = fee_rate + slippage_pct
            
            # Add columns to results
            asset_results['portfolio_value'] = 0.0
            asset_results[f'{asset_id}_holdings_qty'] = 0.0
            asset_results[f'{asset_id}_holdings_value'] = 0.0
            
            # Process each row for trading
            for date, row in asset_results.iterrows():
                # Get prices
                if 'open' in signal_df.columns and date in signal_df.index:
                    open_price = signal_df.loc[date, 'open']
                    close_price = signal_df.loc[date, 'close']
                else:
                    self.logger.warning(f"Missing price data for {asset_id} on {date}, skipping day")
                    continue
                
                # Get decision (from event tracker)
                signal = row['event_decision']
                
                # Track if we executed a trade
                trade_executed = False
                
                # Buy signal and not currently holding
                if signal == 1 and current_holdings_qty == 0:
                    trade_id += 1
                    
                    # Calculate transaction cost
                    transaction_cost = current_cash * total_cost_rate
                    cash_for_purchase = current_cash - transaction_cost
                    
                    # Buy with all available cash
                    quantity = cash_for_purchase / open_price
                    
                    # Record entry
                    current_cash = 0
                    current_holdings_qty = quantity
                    last_entry_price = open_price
                    last_entry_date = date
                    
                    # Record trade
                    entry_trade = {
                        'trade_id': trade_id,
                        'asset': asset_id,
                        'entry_date': date,
                        'exit_date': None,
                        'holding_days': 0,
                        'entry_price': open_price,
                        'exit_price': None,
                        'price_return': "0.00%",
                        'entry_value': cash_for_purchase,
                        'entry_cost': -transaction_cost,
                        'exit_value': None,
                        'exit_cost': None,
                        'trade_return': "0.00%",
                        'is_open': True
                    }
                    
                    asset_trades_list.append(entry_trade)
                    trade_executed = True
                    
                    self.logger.debug(f"{date}: BUY {quantity:.6f} {asset_id} @ {open_price:.2f}")
                
                # Sell signal and currently holding
                elif signal == 0 and current_holdings_qty > 0:
                    # Calculate holding period
                    if last_entry_date:
                        holding_days = (date - last_entry_date).days
                    else:
                        holding_days = 0
                    
                    # Calculate exit value
                    exit_value = current_holdings_qty * open_price
                    
                    # Calculate transaction cost
                    transaction_cost = exit_value * total_cost_rate
                    net_exit_value = exit_value - transaction_cost
                    
                    # Calculate returns
                    price_return_pct = ((open_price / last_entry_price) - 1) * 100
                    
                    # Find the open trade to update
                    for trade in asset_trades_list:
                        if trade['is_open']:
                            # Update exit info
                            trade['exit_date'] = date
                            trade['holding_days'] = holding_days
                            trade['exit_price'] = open_price
                            trade['price_return'] = f"{price_return_pct:.2f}%"
                            trade['exit_value'] = exit_value
                            trade['exit_cost'] = -transaction_cost
                            
                            # Calculate overall trade return
                            entry_value = trade['entry_value']
                            trade_return_pct = ((net_exit_value / entry_value) - 1) * 100
                            trade['trade_return'] = f"{trade_return_pct:.2f}%"
                            trade['is_open'] = False
                            break
                    
                    # Record exit
                    current_cash = net_exit_value
                    current_holdings_qty = 0
                    last_entry_price = 0
                    last_entry_date = None
                    
                    trade_executed = True
                    
                    self.logger.debug(f"{date}: SELL {asset_id} @ {open_price:.2f}, return: {price_return_pct:.2f}%")
                
                # Calculate holdings value
                holdings_value = current_holdings_qty * close_price
                
                # Calculate portfolio value
                portfolio_value = current_cash + holdings_value
                
                # Store in results
                asset_results.loc[date, f'{asset_id}_holdings_qty'] = current_holdings_qty
                asset_results.loc[date, f'{asset_id}_holdings_value'] = holdings_value
                asset_results.loc[date, 'portfolio_value'] = portfolio_value
            
            # Close any open trades at the end of the simulation
            if current_holdings_qty > 0:
                final_date = asset_results.index[-1]
                final_close = signal_df.loc[final_date, 'close']
                
                # Find open trade to update
                for trade in asset_trades_list:
                    if trade['is_open']:
                        # Calculate holding period
                        if last_entry_date:
                            holding_days = (final_date - last_entry_date).days
                        else:
                            holding_days = 0
                        
                        # Update exit info
                        trade['exit_date'] = final_date
                        trade['holding_days'] = holding_days
                        trade['exit_price'] = final_close
                        
                        # Calculate returns
                        price_return_pct = ((final_close / last_entry_price) - 1) * 100
                        trade['price_return'] = f"{price_return_pct:.2f}%"
                        
                        # Calculate exit value
                        exit_value = current_holdings_qty * final_close
                        transaction_cost = exit_value * total_cost_rate
                        net_exit_value = exit_value - transaction_cost
                        
                        trade['exit_value'] = exit_value
                        trade['exit_cost'] = -transaction_cost
                        
                        # Calculate overall trade return
                        entry_value = trade['entry_value']
                        trade_return_pct = ((net_exit_value / entry_value) - 1) * 100
                        trade['trade_return'] = f"{trade_return_pct:.2f}%"
                        
                        # Mark as closed
                        trade['is_open'] = False
                        break
            
            # Calculate daily returns
            asset_results['daily_returns'] = asset_results['portfolio_value'].pct_change()
            asset_results['daily_returns'].iloc[0] = 0.0
            
            # Calculate cumulative returns
            asset_results['cumulative_returns'] = (1 + asset_results['daily_returns']).cumprod() - 1
            
            # Calculate buy-and-hold comparison
            first_valid_open = signal_df['open'].iloc[0]
            
            if first_valid_open > 0:
                buy_hold_qty = initial_capital / first_valid_open
                asset_results['buy_hold_value'] = buy_hold_qty * signal_df['close']
                asset_results['buy_hold_daily_returns'] = asset_results['buy_hold_value'].pct_change()
                asset_results['buy_hold_daily_returns'].iloc[0] = 0.0
                asset_results['buy_hold_cumulative_returns'] = (1 + asset_results['buy_hold_daily_returns']).cumprod() - 1
            
            # Calculate performance metrics
            asset_metrics = self._calculate_asset_metrics(asset_results, asset_trades_list, initial_capital)
            
            # Convert trades list to DataFrame
            asset_trades_df = pd.DataFrame(asset_trades_list)
            
            # Create basket selection series (1 if holding, 0 if not)
            asset_selection = (asset_results[f'{asset_id}_holdings_qty'] > 0).astype(int)
            
            # Store results - only keep specified columns
            results[asset_id] = asset_results[['portfolio_value', 'daily_returns', 'cumulative_returns', 
                                             'buy_hold_value', 'buy_hold_daily_returns', 'buy_hold_cumulative_returns']].reset_index().rename(columns={
                'portfolio_value': 'strategy_value',
                'daily_returns': 'strategy_daily_returns',
                'cumulative_returns': 'strategy_cumulative_returns'
            })
            
            trades[asset_id] = asset_trades_df
            metrics[asset_id] = asset_metrics
            
            # Add this asset's selection to the basket_selection DataFrame
            if basket_selection.empty:
                basket_selection = pd.DataFrame(index=asset_results.index)
            
            basket_selection[asset_id] = asset_selection
        
        # If no assets were processed successfully
        if not results:
            self.logger.error("No assets could be processed successfully in backtest")
            return None
        
        # Finalize basket selection
        basket_selection = basket_selection.reset_index()
        
        # Get event history
        event_history = event_tracker.get_event_history_df()
        
        # Package and return results
        backtest_results = {
            'results': results,
            'signals': signals,
            'trades': trades,
            'metrics': metrics,
            'basket_selection': basket_selection,
            'event_history': event_history,
            'asset_metadata': asset_metadata
        }
        
        self.logger.info(f"Completed event-driven backtest for portfolio '{portfolio_name}' with {len(results)} assets")
        
        return backtest_results
    
    def compare_backtest_methods(
        self,
        portfolio_name: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        initial_capital: float = 10000.0,
        btc_cost: float = 0.001,
        alt_cost: float = 0.005,
        slippage_pct: float = 0.0
    ) -> Dict[str, Any]:
        """Compare traditional binary logic with event-driven logic backtests.
        
        Args:
            portfolio_name: Name of the portfolio to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for backtesting
            btc_cost: Fee rate for Bitcoin transactions
            alt_cost: Fee rate for altcoin transactions
            slippage_pct: Slippage percentage for trading
            
        Returns:
            dict: Dictionary with comparison results
        """
        # Run traditional binary backtest (using the original backtester)
        binary_results = self.portfolio_manager.backtest_portfolio(
            portfolio_name, start_date, end_date, 
            initial_capital, btc_cost, alt_cost, slippage_pct
        )
        
        # Run event-driven backtest
        event_results = self.backtest_portfolio(
            portfolio_name, start_date, end_date, 
            initial_capital, btc_cost, alt_cost, slippage_pct
        )
        
        # If either backtest failed, return None
        if binary_results is None or event_results is None:
            self.logger.error("One or both backtests failed")
            return None
        
        # Create comparison metrics
        comparison = {}
        
        # Compare overall portfolio metrics
        for asset_id in binary_results['metrics']:
            if asset_id in event_results['metrics']:
                binary_metrics = binary_results['metrics'][asset_id]
                event_metrics = event_results['metrics'][asset_id]
                
                # Create side-by-side comparison
                comparison[asset_id] = pd.DataFrame({
                    'Binary Logic': binary_metrics['Strategy'],
                    'Event-Driven': event_metrics['Strategy'],
                    'Difference': event_metrics['Strategy'] - binary_metrics['Strategy'],
                    'Pct Improvement': (event_metrics['Strategy'] - binary_metrics['Strategy']) / 
                                      binary_metrics['Strategy'].abs().replace(0, np.nan) * 100
                })
        
        # Add trade comparison
        binary_trade_count = sum(len(trades) for trades in binary_results['trades'].values() if not trades.empty)
        event_trade_count = sum(len(trades) for trades in event_results['trades'].values() if not trades.empty)
        
        comparison['trade_summary'] = {
            'binary_trades': binary_trade_count,
            'event_driven_trades': event_trade_count,
            'trade_reduction_pct': (binary_trade_count - event_trade_count) / binary_trade_count * 100
                if binary_trade_count > 0 else 0
        }
        
        # Track overall portfolio performance
        binary_portfolio_return = np.mean([
            metrics['Strategy']['Total Return (%)'] 
            for asset_id, metrics in binary_results['metrics'].items()
        ])
        
        event_portfolio_return = np.mean([
            metrics['Strategy']['Total Return (%)'] 
            for asset_id, metrics in event_results['metrics'].items()
        ])
        
        comparison['portfolio_summary'] = {
            'binary_return': binary_portfolio_return,
            'event_driven_return': event_portfolio_return,
            'return_improvement_pct': ((event_portfolio_return - binary_portfolio_return) / 
                                     abs(binary_portfolio_return)) * 100
                if binary_portfolio_return != 0 else 0
        }
        
        self.logger.info(
            f"Comparison complete: Event-driven approach {'improved' if comparison['portfolio_summary']['return_improvement_pct'] > 0 else 'reduced'} "
            f"returns by {abs(comparison['portfolio_summary']['return_improvement_pct']):.2f}% "
            f"and {'reduced' if comparison['trade_summary']['trade_reduction_pct'] > 0 else 'increased'} "
            f"trades by {abs(comparison['trade_summary']['trade_reduction_pct']):.2f}%"
        )
        
        return {
            'comparison': comparison,
            'binary_results': binary_results,
            'event_results': event_results
        }
    
    def _calculate_asset_metrics(self, asset_results, asset_trades_list, initial_capital):
        """Calculate performance metrics for an asset's simulation.
        
        Args:
            asset_results: DataFrame with daily results
            asset_trades_list: List of trade dictionaries
            initial_capital: Initial capital
            
        Returns:
            DataFrame: DataFrame with Strategy and Buy & Hold metrics
        """
        metrics_dict = {}
        
        # Basic return metrics
        final_value = asset_results['portfolio_value'].iloc[-1]
        metrics_dict['Total Return (%)'] = ((final_value / initial_capital) - 1) * 100
        
        # Max drawdown
        # The max_drawdown method expects cumulative returns + 1 (i.e., growth factor)
        if 'cum_returns_plus_one' not in asset_results.columns:
            asset_results['cum_returns_plus_one'] = asset_results['cumulative_returns'] + 1
        
        metrics_dict['Max Drawdown (%)'] = self.metrics_calculator.max_drawdown(asset_results['cum_returns_plus_one']) * 100
        
        # Sharpe & Sortino ratios
        if not asset_results['daily_returns'].empty:
            sharpe, sortino, avg_daily_return, _ = self.metrics_calculator.calculate_sharpe_sortino(
                asset_results, 'daily_returns'
            )
            metrics_dict['Sharpe Ratio'] = sharpe
            metrics_dict['Sortino Ratio'] = sortino
            
            # Annualized metrics
            trading_days_per_year = 365
            daily_std_dev = asset_results['daily_returns'].std()
            annualized_volatility = daily_std_dev * np.sqrt(trading_days_per_year)
            metrics_dict['Annualized Volatility (%)'] = annualized_volatility * 100
            
            if pd.notna(avg_daily_return):
                annualized_return = ((1 + avg_daily_return) ** trading_days_per_year) - 1
                metrics_dict['Annualized Return (%)'] = annualized_return * 100
            else:
                metrics_dict['Annualized Return (%)'] = 0.0
        else:
            metrics_dict['Sharpe Ratio'] = 0.0
            metrics_dict['Sortino Ratio'] = 0.0
            metrics_dict['Annualized Volatility (%)'] = 0.0
            metrics_dict['Annualized Return (%)'] = 0.0
        
        # Trade metrics
        metrics_dict['Total Trades'] = len([t for t in asset_trades_list if not t['is_open']])
        
        # Calculate win rate and avg win/loss only for closed trades
        closed_trades = [t for t in asset_trades_list if not t['is_open']]
        
        if closed_trades:
            # Parse trade returns
            for trade in closed_trades:
                if isinstance(trade['trade_return'], str) and trade['trade_return'].endswith('%'):
                    trade['trade_return_pct'] = float(trade['trade_return'].rstrip('%'))
                else:
                    trade['trade_return_pct'] = 0.0
            
            # Calculate win rate
            winning_trades = [t for t in closed_trades if t['trade_return_pct'] > 0]
            win_rate = (len(winning_trades) / len(closed_trades)) * 100
            metrics_dict['Win Rate (%)'] = win_rate
            
            # Calculate average win and loss
            if winning_trades:
                avg_win = np.mean([t['trade_return_pct'] for t in winning_trades])
                metrics_dict['Avg Win (%)'] = avg_win
            else:
                metrics_dict['Avg Win (%)'] = 0.0
            
            losing_trades = [t for t in closed_trades if t['trade_return_pct'] <= 0]
            if losing_trades:
                avg_loss = np.mean([t['trade_return_pct'] for t in losing_trades])
                metrics_dict['Avg Loss (%)'] = avg_loss
            else:
                metrics_dict['Avg Loss (%)'] = 0.0
            
            # Calculate profit factor
            total_wins = sum([t['trade_return_pct'] for t in winning_trades])
            total_losses = abs(sum([t['trade_return_pct'] for t in losing_trades]))
            
            if total_losses > 0:
                profit_factor = total_wins / total_losses
            else:
                profit_factor = total_wins if total_wins > 0 else 0.0
            
            metrics_dict['Profit Factor'] = profit_factor
        else:
            metrics_dict['Win Rate (%)'] = 0.0
            metrics_dict['Avg Win (%)'] = 0.0
            metrics_dict['Avg Loss (%)'] = 0.0
            metrics_dict['Profit Factor'] = 0.0
        
        # Calculate transaction costs
        total_costs = sum(abs(t.get('entry_cost', 0)) for t in asset_trades_list)
        total_costs += sum(abs(t.get('exit_cost', 0)) for t in asset_trades_list if not t['is_open'])
        metrics_dict['Total Transaction Costs'] = total_costs
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Strategy': [
                metrics_dict['Total Return (%)'],
                metrics_dict['Max Drawdown (%)'],
                metrics_dict['Sharpe Ratio'],
                metrics_dict['Sortino Ratio'],
                metrics_dict['Annualized Volatility (%)'],
                metrics_dict['Annualized Return (%)'],
                metrics_dict['Total Trades'],
                metrics_dict['Win Rate (%)'],
                metrics_dict['Avg Win (%)'],
                metrics_dict['Avg Loss (%)'],
                metrics_dict['Profit Factor'],
                metrics_dict['Total Transaction Costs']
            ],
            'Buy & Hold': [
                metrics_dict.get('Buy & Hold Return (%)', 0.0),
                metrics_dict.get('Buy & Hold Max Drawdown (%)', 0.0),
                metrics_dict.get('Buy & Hold Sharpe Ratio', 0.0),
                metrics_dict.get('Buy & Hold Sortino Ratio', 0.0),
                metrics_dict.get('Buy & Hold Annualized Volatility (%)', 0.0),
                metrics_dict.get('Buy & Hold Annualized Return (%)', 0.0),
                '-',  # No trades for buy & hold
                '-',  # No win rate for buy & hold
                '-',  # No avg win for buy & hold
                '-',  # No avg loss for buy & hold
                '-',  # No profit factor for buy & hold
                0.0   # No transaction costs for buy & hold
            ]
        }, index=[
            'Total Return (%)',
            'Max Drawdown (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Annualized Volatility (%)',
            'Annualized Return (%)',
            'Total Trades',
            'Win Rate (%)',
            'Avg Win (%)',
            'Avg Loss (%)',
            'Profit Factor',
            'Total Transaction Costs'
        ])
        
        return metrics_df
    
    def backtest_assets(
        self,
        asset_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0,
        trade_cost: float = 0.001,
        slippage_pct: float = 0.0,
        signal_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Backtest multiple assets.
        
        Args:
            asset_ids: List of asset IDs to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for each asset
            trade_cost: Trading cost as percentage
            slippage_pct: Slippage percentage
            signal_threshold: Signal threshold for decisions
            
        Returns:
            dict: Dictionary with backtest results
        """
        self.logger.info(f"Backtesting {len(asset_ids)} assets...")
        
        # Initialize results
        asset_results = {}
        overall_results = {}
        
        # Process each asset
        for asset_id in asset_ids:
            try:
                # Get price data
                if self.analyzer:
                    asset_data = self.analyzer.get_asset_data(asset_id)
                    if asset_data is None:
                        self.logger.warning(f"Asset data not found for {asset_id}, skipping asset")
                        continue
                    price_data = asset_data.price_data
                else:
                    price_data = self.data_loader.get_asset_data(asset_id)
                    
                if price_data is None or price_data.empty:
                    self.logger.warning(f"No price data for {asset_id}, skipping asset")
                    continue
                
                # Filter date range if specified
                if start_date is not None or end_date is not None:
                    # Handle start_date filtering
                    if start_date is not None:
                        price_data = price_data.loc[price_data.index >= start_date]
                    
                    # Handle end_date filtering    
                    if end_date is not None:
                        price_data = price_data.loc[price_data.index <= end_date]
                
                if price_data.empty:
                    self.logger.warning(f"No data in date range for {asset_id}, skipping asset")
                    continue
                    
                # Get signal events
                if self.analyzer:
                    events = self.analyzer.signal_event_tracker.get_events(asset_id)
                else:
                    events = self.signal_events
                    
                if events is None or events.empty:
                    self.logger.warning(f"No signal events for {asset_id}, skipping asset")
                    continue
                    
                # Run backtest
                self.signal_events = events
                self.price_data = price_data
                self.asset_id = asset_id
                self.initial_capital = initial_capital
                self.trade_cost = trade_cost
                self.slippage_pct = slippage_pct
                self.signal_threshold = signal_threshold
                
                result = self.run()
                asset_results[asset_id] = result
                
                self.logger.info(
                    f"Backtest for {asset_id}: "
                    f"Return: {result['total_return']:.2%}, "
                    f"Sharpe: {result['sharpe_ratio']:.2f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error backtesting {asset_id}: {str(e)}")
                if self.debug:
                    import traceback
                    self.logger.error(traceback.format_exc())
                continue
                
        # Calculate overall results
        if asset_results:
            # Combine results
            overall_results = self._combine_results(asset_results)
            
        return {
            'asset_results': asset_results,
            'overall_results': overall_results
        } 