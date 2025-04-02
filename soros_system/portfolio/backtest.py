import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from ..analysis.metrics import MetricsCalculator
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Class for backtesting portfolio strategies by simulating trades for each asset independently.
    """

    def __init__(self, portfolio_manager, data_loader):
        """
        Initialize the PortfolioBacktester.
        
        Args:
            portfolio_manager: Manager for portfolios
            data_loader: Loader for asset data
        """
        self.portfolio_manager = portfolio_manager
        self.data_loader = data_loader
        self.metrics_calculator = MetricsCalculator()
        self.logger = logging.getLogger(__name__)

    def backtest_portfolio(self,
                           portfolio_name,
                           start_date,
                           end_date,
                           initial_capital=10000.0,
                           btc_cost=0.001,
                           alt_cost=0.005,
                           slippage_pct=0.0):
        """
        Backtest a portfolio strategy by simulating trades for each asset independently with full cash.
        
        Args:
            portfolio_name (str): Name of the portfolio to backtest
            start_date (str or datetime): Start date for backtesting
            end_date (str or datetime): End date for backtesting
            initial_capital (float): Initial capital for backtesting (default: 10000.0)
            btc_cost (float): Fee rate for Bitcoin transactions (default: 0.001)
            alt_cost (float): Fee rate for altcoin transactions (default: 0.005)
            slippage_pct (float): Slippage percentage for trading (default: 0.0)
            
        Returns:
            dict: Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest for portfolio '{portfolio_name}' from {start_date} to {end_date}")
        
        # Convert dates to datetime
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return None
        
        # Get signal tables from portfolio manager
        signal_tables = self.portfolio_manager.get_portfolio_signals(portfolio_name)
        if signal_tables is None:
            self.logger.error(f"Failed to get signal tables for portfolio '{portfolio_name}'")
            return None
        
        if not signal_tables:
            self.logger.error(f"No signal tables available for portfolio '{portfolio_name}'")
            return None
        
        self.logger.info(f"Retrieved signal tables for {len(signal_tables)} assets")
        
        # Initialize results containers
        results = {}
        signals = {}
        trades = {}
        metrics = {}
        basket_selection = pd.DataFrame()
        
        # Process each asset independently
        for asset_id, signal_df in signal_tables.items():
            self.logger.info(f"Simulating trades for {asset_id}")
            
            # Filter to date range
            if 'date' in signal_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(signal_df['date']):
                    signal_df['date'] = pd.to_datetime(signal_df['date'])
                # Filter by date range
                signal_df = signal_df[(signal_df['date'] >= start_date) & (signal_df['date'] <= end_date)]
                # Set index for easier processing
                signal_df = signal_df.set_index('date')
            elif isinstance(signal_df.index, pd.DatetimeIndex):
                # Filter by date range using index
                signal_df = signal_df[(signal_df.index >= start_date) & (signal_df.index <= end_date)]
            else:
                self.logger.error(f"No date column or DatetimeIndex in signal data for {asset_id}")
                continue
            
            if signal_df.empty:
                self.logger.warning(f"No data available for {asset_id} in the specified date range")
                continue
            
            # Check for necessary columns
            required_cols = ['open', 'close', 'final_decision', 'final_decision_shifted']
            missing_cols = [col for col in required_cols if col not in signal_df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns for {asset_id}: {missing_cols}")
                # Try to fill missing price columns if possible
                if 'open' in missing_cols and 'close' in signal_df.columns:
                    self.logger.info(f"Using 'close' for 'open' in {asset_id}")
                    signal_df['open'] = signal_df['close']
                    missing_cols.remove('open')
                
                if 'close' in missing_cols and 'open' in signal_df.columns:
                    self.logger.info(f"Using 'open' for 'close' in {asset_id}")
                    signal_df['close'] = signal_df['open']
                    missing_cols.remove('close')
                
                if 'final_decision_shifted' in missing_cols:
                    self.logger.error(f"Missing critical signal column 'final_decision_shifted' for {asset_id}")
                    continue
                
                if 'final_decision' in missing_cols:
                    self.logger.warning(f"Missing control signal column 'final_decision' for {asset_id}")
                    # Create final_decision from final_decision_shifted for control purposes
                    signal_df['final_decision'] = signal_df['final_decision_shifted'].shift(-1)
                    missing_cols.remove('final_decision')
            
            # Fill missing prices with forward fill
            if signal_df[['open', 'close']].isnull().any().any():
                self.logger.warning(f"Filling missing prices for {asset_id} with forward fill")
                signal_df[['open', 'close']] = signal_df[['open', 'close']].ffill()
            
            # Drop rows with remaining NaN prices
            initial_len = len(signal_df)
            signal_df = signal_df.dropna(subset=['open', 'close'])
            if len(signal_df) < initial_len:
                self.logger.warning(f"Dropped {initial_len - len(signal_df)} rows with NaN prices for {asset_id}")
            
            if signal_df.empty:
                self.logger.warning(f"No data left after dropping rows with NaN prices for {asset_id}")
                continue
            
            # Ensure numeric price columns
            signal_df[['open', 'close']] = signal_df[['open', 'close']].apply(pd.to_numeric, errors='coerce')
            
            # Create results dataframe
            asset_results = pd.DataFrame(index=signal_df.index)
            
            # Simulate trades for this asset with full initial capital
            asset_trades_list = []
            
            # Initial state
            current_cash = initial_capital
            current_holdings_qty = 0.0
            last_entry_price = 0.0
            last_entry_date = None
            trade_id = 0
            
            # Date-sorted index for chronological processing
            for date in signal_df.index:
                # Get data for this date
                row = signal_df.loc[date]
                open_price = row['open']
                close_price = row['close']
                signal = row['final_decision_shifted']
                
                # Skip first day if signal is NaN
                if pd.isna(signal):
                    asset_results.loc[date, 'cash'] = current_cash
                    asset_results.loc[date, f'{asset_id}_holdings_qty'] = 0.0
                    asset_results.loc[date, f'{asset_id}_holdings_value'] = 0.0
                    asset_results.loc[date, 'portfolio_value'] = current_cash
                    continue
                
                # Skip if invalid price
                if open_price <= 0:
                    self.logger.warning(f"Invalid price ({open_price}) for {asset_id} on {date}. Skipping.")
                    # Maintain current state
                    asset_results.loc[date, 'cash'] = current_cash
                    asset_results.loc[date, f'{asset_id}_holdings_qty'] = current_holdings_qty
                    asset_results.loc[date, f'{asset_id}_holdings_value'] = current_holdings_qty * close_price if close_price > 0 else 0
                    asset_results.loc[date, 'portfolio_value'] = current_cash + asset_results.loc[date, f'{asset_id}_holdings_value']
                    continue
                
                # Determine fee rate based on asset
                fee_rate = btc_cost if asset_id == 'bitcoin' else alt_cost
                total_cost_rate = fee_rate + slippage_pct
                
                # Trading logic
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
                    # Calculate gross value and transaction cost
                    gross_value = current_holdings_qty * open_price
                    transaction_cost = gross_value * total_cost_rate
                    net_value = gross_value - transaction_cost
                    
                    # Calculate returns
                    if last_entry_price > 0:
                        price_return_pct = (open_price / last_entry_price - 1) * 100
                        price_return_str = f"{price_return_pct:.2f}%"
                        
                        # Calculate holding period
                        if last_entry_date:
                            holding_days = (date - last_entry_date).days
                        else:
                            holding_days = 0
                        
                        # Find the entry trade to update
                        for trade in asset_trades_list:
                            if trade['is_open']:
                                # Update exit info
                                trade['exit_date'] = date
                                trade['holding_days'] = holding_days
                                trade['exit_price'] = open_price
                                trade['price_return'] = price_return_str
                                trade['exit_value'] = net_value
                                trade['exit_cost'] = -transaction_cost
                                
                                # Calculate net trade return (including costs)
                                entry_value = trade['entry_value']
                                entry_cost = trade['entry_cost']
                                
                                # Net return calculation including costs - FIXED
                                # Use entry_value as the base for percentage calculation
                                trade_return_pct = ((net_value - entry_value) / entry_value) * 100
                                trade['trade_return'] = f"{trade_return_pct:.2f}%"
                                trade['is_open'] = False
                                break
                    
                    # Execute sell
                    current_cash = net_value
                    current_holdings_qty = 0
                    last_entry_price = 0
                    last_entry_date = None
                    trade_executed = True
                    
                    self.logger.debug(f"{date}: SELL @ {open_price:.2f}, received {net_value:.2f}")
                
                # Update portfolio value
                holdings_value = current_holdings_qty * close_price
                portfolio_value = current_cash + holdings_value
                
                # Store daily results
                asset_results.loc[date, 'cash'] = current_cash
                asset_results.loc[date, f'{asset_id}_holdings_qty'] = current_holdings_qty
                asset_results.loc[date, f'{asset_id}_holdings_value'] = holdings_value
                asset_results.loc[date, 'portfolio_value'] = portfolio_value
            
            # Close any open trades at the end of the simulation
            if current_holdings_qty > 0:
                final_date = signal_df.index[-1]
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
                        if last_entry_price > 0:
                            price_return_pct = (final_close / last_entry_price - 1) * 100
                            trade['price_return'] = f"{price_return_pct:.2f}%"
                            
                            # Calculate gross value and transaction cost
                            gross_value = current_holdings_qty * final_close
                            transaction_cost = gross_value * total_cost_rate
                            net_value = gross_value - transaction_cost
                            
                            trade['exit_value'] = net_value
                            trade['exit_cost'] = -transaction_cost
                            
                            # Calculate net trade return (including costs)
                            entry_value = trade['entry_value']
                            entry_cost = trade['entry_cost']
                            
                            # Net return calculation including costs - FIXED
                            # Use entry_value as the base for percentage calculation
                            trade_return_pct = ((net_value - entry_value) / entry_value) * 100
                            trade['trade_return'] = f"{trade_return_pct:.2f}%"
                            
                            # Mark trade as "still open at end of simulation"
                            trade['is_open'] = True
                            break
            
            # Calculate daily returns
            asset_results['daily_returns'] = asset_results['portfolio_value'].pct_change()
            asset_results['daily_returns'].iloc[0] = 0.0  # First day
            asset_results['cumulative_returns'] = (1 + asset_results['daily_returns']).cumprod() - 1
            
            # Calculate buy & hold benchmark
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
            
            signals[asset_id] = signal_df.reset_index()
            trades[asset_id] = asset_trades_df
            metrics[asset_id] = asset_metrics  # Store the metrics DataFrame directly
            
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
        
        # Package and return results
        backtest_results = {
            'results': results,
            'signals': signals,
            'trades': trades,
            'metrics': metrics,
            'basket_selection': basket_selection
        }
        
        self.logger.info(f"Completed backtest for portfolio '{portfolio_name}' with {len(results)} assets")
        
        return backtest_results
    
    def _calculate_asset_metrics(self, asset_results, asset_trades_list, initial_capital):
        """
        Calculate performance metrics for an asset's simulation.
        
        Args:
            asset_results (pd.DataFrame): DataFrame with daily results
            asset_trades_list (list): List of trade dictionaries
            initial_capital (float): Initial capital
            
        Returns:
            pd.DataFrame: DataFrame with Strategy and Buy & Hold metrics
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
            # Extract trade returns from percentage strings
            trade_returns = [float(t['trade_return'].replace('%', '')) for t in closed_trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r <= 0]
            
            metrics_dict['Win Rate (%)'] = (len(winning_trades) / len(trade_returns)) * 100 if trade_returns else 0
            metrics_dict['Avg Win (%)'] = np.mean(winning_trades) if winning_trades else 0
            metrics_dict['Avg Loss (%)'] = np.mean(losing_trades) if losing_trades else 0
            metrics_dict['Profit Factor'] = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else 0
        else:
            metrics_dict['Win Rate (%)'] = 0
            metrics_dict['Avg Win (%)'] = 0
            metrics_dict['Avg Loss (%)'] = 0
            metrics_dict['Profit Factor'] = 0
        
        # Transaction costs
        entry_costs = sum([t['entry_cost'] for t in asset_trades_list])
        exit_costs = sum([t['exit_cost'] for t in asset_trades_list if t['exit_cost'] is not None])
        metrics_dict['Total Transaction Costs'] = abs(entry_costs) + abs(exit_costs)
        
        # Buy & Hold metrics
        if 'buy_hold_value' in asset_results.columns and not asset_results['buy_hold_value'].empty:
            buy_hold_final_value = asset_results['buy_hold_value'].iloc[-1]
            metrics_dict['Buy & Hold Return (%)'] = ((buy_hold_final_value / initial_capital) - 1) * 100
            
            # Calculate buy & hold max drawdown
            if 'buy_hold_cumulative_returns' in asset_results.columns:
                buy_hold_cum_returns_plus_one = asset_results['buy_hold_cumulative_returns'] + 1
                metrics_dict['Buy & Hold Max Drawdown (%)'] = self.metrics_calculator.max_drawdown(buy_hold_cum_returns_plus_one) * 100
            
            # Calculate buy & hold Sharpe and Sortino ratios
            if 'buy_hold_daily_returns' in asset_results.columns and not asset_results['buy_hold_daily_returns'].empty:
                buy_hold_sharpe, buy_hold_sortino, buy_hold_avg_daily_return, _ = self.metrics_calculator.calculate_sharpe_sortino(
                    asset_results, 'buy_hold_daily_returns'
                )
                metrics_dict['Buy & Hold Sharpe Ratio'] = buy_hold_sharpe
                metrics_dict['Buy & Hold Sortino Ratio'] = buy_hold_sortino
                
                # Buy & Hold annualized metrics
                buy_hold_daily_std_dev = asset_results['buy_hold_daily_returns'].std()
                buy_hold_annualized_volatility = buy_hold_daily_std_dev * np.sqrt(trading_days_per_year)
                metrics_dict['Buy & Hold Annualized Volatility (%)'] = buy_hold_annualized_volatility * 100
                
                if pd.notna(buy_hold_avg_daily_return):
                    buy_hold_annualized_return = ((1 + buy_hold_avg_daily_return) ** trading_days_per_year) - 1
                    metrics_dict['Buy & Hold Annualized Return (%)'] = buy_hold_annualized_return * 100
                else:
                    metrics_dict['Buy & Hold Annualized Return (%)'] = 0.0
            else:
                metrics_dict['Buy & Hold Sharpe Ratio'] = 0.0
                metrics_dict['Buy & Hold Sortino Ratio'] = 0.0
                metrics_dict['Buy & Hold Annualized Volatility (%)'] = 0.0
                metrics_dict['Buy & Hold Annualized Return (%)'] = 0.0
        else:
            metrics_dict['Buy & Hold Return (%)'] = 0.0
            metrics_dict['Buy & Hold Max Drawdown (%)'] = 0.0
            metrics_dict['Buy & Hold Sharpe Ratio'] = 0.0
            metrics_dict['Buy & Hold Sortino Ratio'] = 0.0
            metrics_dict['Buy & Hold Annualized Volatility (%)'] = 0.0
            metrics_dict['Buy & Hold Annualized Return (%)'] = 0.0
        
        # Create DataFrame with Strategy and Buy & Hold columns
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
                metrics_dict['Buy & Hold Return (%)'],
                metrics_dict['Buy & Hold Max Drawdown (%)'],
                metrics_dict['Buy & Hold Sharpe Ratio'],
                metrics_dict['Buy & Hold Sortino Ratio'],
                metrics_dict['Buy & Hold Annualized Volatility (%)'],
                metrics_dict['Buy & Hold Annualized Return (%)'],
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
