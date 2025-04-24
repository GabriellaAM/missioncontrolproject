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

    def backtest_asset(self,
                      asset_id,
                      start_date,
                      end_date,
                      initial_capital=10000.0,
                      trade_cost=0.001,
                      slippage_pct=0.0,
                      usd_signals=None,
                      btc_signals=None):
        """
        Backtest a trading strategy for a single asset using specified signals.
        
        Args:
            asset_id (str): Asset ID to backtest
            start_date (str or datetime): Start date for backtesting
            end_date (str or datetime): End date for backtesting
            initial_capital (float): Initial capital for backtesting (default: 10000.0)
            trade_cost (float): Fee rate for transactions (default: 0.001)
            slippage_pct (float): Slippage percentage for trading (default: 0.0)
            usd_signals (list): List of USD signal names to use (default: None)
            btc_signals (list): List of BTC signal names to use (default: None)
            
        Returns:
            dict: Dictionary with backtest results, including dataframe, evaluation metrics, and summary
        """
        self.logger.info(f"Starting backtest for asset '{asset_id}' from {start_date} to {end_date}")
        
        # Validate input signals
        if (usd_signals is None or len(usd_signals) == 0) and (btc_signals is None or len(btc_signals) == 0):
            raise ValueError("At least one of usd_signals or btc_signals must be provided")
        
        # For Bitcoin, only USD signals are valid
        if asset_id == 'bitcoin' and btc_signals and len(btc_signals) > 0:
            self.logger.warning("BTC signals are not applicable for Bitcoin itself. Using only USD signals.")
            btc_signals = None
            
        # Convert dates to datetime
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return None
        
        # Load asset data - corrected for PortfolioAnalyzer
        asset_data = None
        if hasattr(self.data_loader, 'assets'):
            # Direct access to PortfolioAnalyzer's assets dictionary
            asset_obj = self.data_loader.assets.get(asset_id)
            if asset_obj and hasattr(asset_obj, 'price_data'):
                asset_data = asset_obj.price_data
        elif hasattr(self.data_loader, 'get_asset_data'):
            # Using PortfolioAnalyzer's get_asset_data method
            asset = self.data_loader.get_asset_data(asset_id)
            if asset and hasattr(asset, 'price_data'):
                asset_data = asset.price_data
        elif hasattr(self.data_loader, 'get_asset_processed_data'):
            # Fallback to TrendAnalyzer method
            asset_data = self.data_loader.get_asset_processed_data(asset_id)
            
        if asset_data is None or asset_data.empty:
            self.logger.error(f"Failed to load data for asset '{asset_id}'")
            return None
        
        # Ensure we have a DatetimeIndex
        if not isinstance(asset_data.index, pd.DatetimeIndex) and 'date' in asset_data.columns:
            asset_data['date'] = pd.to_datetime(asset_data['date'])
            asset_data = asset_data.set_index('date')
        
        # Filter to date range
        asset_data = asset_data[(asset_data.index >= start_date) & (asset_data.index <= end_date)]
        
        if asset_data.empty:
            self.logger.warning(f"No data available for {asset_id} in the specified date range")
            return None
        
        # Check for required price columns
        required_cols = ['open', 'close']
        missing_cols = [col for col in required_cols if col not in asset_data.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required price columns for {asset_id}: {missing_cols}")
            # Try to fill missing price columns
            if 'open' in missing_cols and 'close' in asset_data.columns:
                self.logger.info(f"Using 'close' for 'open' in {asset_id}")
                asset_data['open'] = asset_data['close']
                missing_cols.remove('open')
            
            if 'close' in missing_cols and 'open' in asset_data.columns:
                self.logger.info(f"Using 'open' for 'close' in {asset_id}")
                asset_data['close'] = asset_data['open']
                missing_cols.remove('close')
            
            if missing_cols:
                self.logger.error(f"Still missing critical price columns for {asset_id}: {missing_cols}")
                return None
        
        # Ensure we have signal values for each specified signal
        all_signals = []
        if usd_signals:
            all_signals.extend(usd_signals)
        if btc_signals:
            all_signals.extend(btc_signals)
            
        # Get signal values
        signal_values = {}
        missing_signals = []
        
        for signal_name in all_signals:
            # Check if signal is already in asset_data
            if signal_name in asset_data.columns:
                signal_values[signal_name] = asset_data[signal_name]
            else:
                # Try to get signal from asset in the analyzer
                signal_found = False
                
                # Try direct access to assets dictionary
                if hasattr(self.data_loader, 'assets'):
                    asset = self.data_loader.assets.get(asset_id)
                    if asset and hasattr(asset, 'get_signal'):
                        signal_data = asset.get_signal(signal_name)
                        if signal_data and hasattr(signal_data, 'values') and signal_data.values is not None:
                            # Ensure the values have the same index as our price data
                            signal_series = pd.Series(
                                signal_data.values, 
                                index=signal_data.values.index if hasattr(signal_data.values, 'index') else None
                            )
                            if hasattr(signal_series, 'reindex'):
                                signal_series = signal_series.reindex(asset_data.index)
                            signal_values[signal_name] = signal_series
                            signal_found = True
                
                # Fallback to get_asset_data
                if not signal_found and hasattr(self.data_loader, 'get_asset_data'):
                    asset = self.data_loader.get_asset_data(asset_id)
                    if asset and hasattr(asset, 'get_signal'):
                        signal_data = asset.get_signal(signal_name)
                        if signal_data and hasattr(signal_data, 'values') and signal_data.values is not None:
                            # Ensure the values have the same index as our price data
                            signal_series = pd.Series(
                                signal_data.values, 
                                index=signal_data.values.index if hasattr(signal_data.values, 'index') else None
                            )
                            if hasattr(signal_series, 'reindex'):
                                signal_series = signal_series.reindex(asset_data.index)
                            signal_values[signal_name] = signal_series
                            signal_found = True
                
                if not signal_found:
                    missing_signals.append(signal_name)
        
        if missing_signals:
            self.logger.warning(f"Could not find signal values for: {missing_signals}")
            all_signals = [s for s in all_signals if s not in missing_signals]
            
            if len(all_signals) == 0:
                self.logger.error("No valid signals available. Cannot proceed with backtest.")
                return None
        
        # Create a DataFrame for our backtest results
        backtest_df = pd.DataFrame(index=asset_data.index)
        
        # Add price data
        backtest_df['open'] = asset_data['open']
        backtest_df['close'] = asset_data['close']
        
        # Add signals
        for signal_name, values in signal_values.items():
            backtest_df[signal_name] = values
        
        # Calculate final signal decision based on USD and BTC signals
        # For BTC, only use USD signals
        if asset_id == 'bitcoin':
            if usd_signals and len(usd_signals) > 0:
                # Calculate USD signal - 1 if all USD signals are 1, 0 otherwise
                backtest_df['usd_signal'] = backtest_df[usd_signals].prod(axis=1)
                backtest_df['final_decision'] = backtest_df['usd_signal']
            else:
                self.logger.error("No USD signals provided for Bitcoin")
                return None
        else:
            # For altcoins, can use both USD and BTC signals
            if usd_signals and len(usd_signals) > 0:
                backtest_df['usd_signal'] = backtest_df[usd_signals].prod(axis=1)
            else:
                backtest_df['usd_signal'] = 1  # Default to 1 if no USD signals
                
            if btc_signals and len(btc_signals) > 0:
                backtest_df['btc_signal'] = backtest_df[btc_signals].prod(axis=1)
            else:
                backtest_df['btc_signal'] = 1  # Default to 1 if no BTC signals
                
            # Final decision is 1 if both USD and BTC signals are 1
            # If only one type of signal is provided, only use that one
            if usd_signals and len(usd_signals) > 0 and btc_signals and len(btc_signals) > 0:
                backtest_df['final_decision'] = (backtest_df['usd_signal'] & backtest_df['btc_signal']).astype(int)
            elif usd_signals and len(usd_signals) > 0:
                backtest_df['final_decision'] = backtest_df['usd_signal']
            elif btc_signals and len(btc_signals) > 0:
                backtest_df['final_decision'] = backtest_df['btc_signal']
        
        # Shift the final decision to avoid lookahead bias 
        # (use today's signal for tomorrow's open trade)
        backtest_df['final_decision_shifted'] = backtest_df['final_decision'].shift(1)
        
        # Initialize trading simulation
        backtest_df['position'] = 0
        backtest_df['cash'] = initial_capital
        backtest_df['holdings_qty'] = 0.0
        backtest_df['holdings_value'] = 0.0
        backtest_df['portfolio_value'] = initial_capital
        backtest_df['trade_executed'] = False
        backtest_df['daily_return'] = 0.0
        backtest_df['strategy_cumulative_return'] = 1.0
        
        # Buy and hold reference values
        buy_price = asset_data['open'].iloc[1]  # Start with second day opening price
        buy_qty = initial_capital / buy_price
        backtest_df['buy_hold_value'] = asset_data['close'] * buy_qty
        backtest_df['buy_hold_daily_return'] = backtest_df['buy_hold_value'].pct_change()
        backtest_df['buy_hold_cumulative_return'] = (1 + backtest_df['buy_hold_daily_return']).cumprod()
        
        # Trading simulation
        current_cash = initial_capital
        current_holdings_qty = 0.0
        total_cost_rate = trade_cost + slippage_pct
        trades = []
        
        # Track position entry details
        entry_price = 0.0
        entry_date = None
        trade_id = 0
        
        # Exclude the first row with NaN values
        for date in backtest_df.index[1:]:
            row = backtest_df.loc[date]
            signal = row['final_decision_shifted']
            open_price = row['open']
            close_price = row['close']
            
            # Skip if signal is NaN or prices are invalid
            if pd.isna(signal) or open_price <= 0:
                backtest_df.loc[date, 'cash'] = current_cash
                backtest_df.loc[date, 'holdings_qty'] = current_holdings_qty
                backtest_df.loc[date, 'holdings_value'] = current_holdings_qty * close_price
                backtest_df.loc[date, 'portfolio_value'] = current_cash + backtest_df.loc[date, 'holdings_value']
                continue
            
            # Flag for trade execution
            trade_executed = False
            
            # Buy signal (1) and not holding
            if signal == 1 and current_holdings_qty == 0:
                trade_id += 1
                
                # Calculate transaction cost
                transaction_cost = current_cash * total_cost_rate
                cash_for_purchase = current_cash - transaction_cost
                
                # Buy with all available cash
                quantity = cash_for_purchase / open_price
                
                # Update holdings
                current_cash = 0
                current_holdings_qty = quantity
                entry_price = open_price
                entry_date = date
                
                # Record trade for later analysis
                entry_trade = {
                    'trade_id': trade_id,
                    'asset': asset_id,
                    'entry_date': date,
                    'exit_date': None,
                    'holding_days': 0,
                    'entry_price': open_price,
                    'exit_price': None,
                    'price_return': 0.0,
                    'entry_value': cash_for_purchase,
                    'entry_cost': transaction_cost,
                    'exit_value': None,
                    'exit_cost': None,
                    'trade_return': 0.0,
                    'is_open': True
                }
                trades.append(entry_trade)
                trade_executed = True
                
                # Mark the position
                backtest_df.loc[date, 'position'] = 1
                
            # Sell signal (0) and currently holding
            elif signal == 0 and current_holdings_qty > 0:
                # Calculate gross value and transaction cost
                gross_value = current_holdings_qty * open_price
                transaction_cost = gross_value * total_cost_rate
                net_value = gross_value - transaction_cost
                
                # Update holdings
                current_cash = net_value
                current_holdings_qty = 0
                
                # Update the exit information for the last entry trade
                holding_days = (date - entry_date).days if entry_date else 0
                price_return = (open_price / entry_price - 1) if entry_price > 0 else 0
                trade_return = (net_value / trades[-1]['entry_value'] - 1) if trades[-1]['entry_value'] > 0 else 0
                
                trades[-1].update({
                    'exit_date': date,
                    'holding_days': holding_days,
                    'exit_price': open_price,
                    'price_return': price_return,
                    'exit_value': gross_value,
                    'exit_cost': transaction_cost,
                    'trade_return': trade_return,
                    'is_open': False
                })
                
                trade_executed = True
                
                # Mark the position
                backtest_df.loc[date, 'position'] = 0
                
            else:
                # No change in position
                backtest_df.loc[date, 'position'] = 1 if current_holdings_qty > 0 else 0
            
            # Update the backtest DataFrame
            backtest_df.loc[date, 'cash'] = current_cash
            backtest_df.loc[date, 'holdings_qty'] = current_holdings_qty
            backtest_df.loc[date, 'holdings_value'] = current_holdings_qty * close_price
            backtest_df.loc[date, 'portfolio_value'] = current_cash + backtest_df.loc[date, 'holdings_value']
            backtest_df.loc[date, 'trade_executed'] = trade_executed
        
        # Calculate strategy returns
        backtest_df['daily_return'] = backtest_df['portfolio_value'].pct_change()
        backtest_df['strategy_cumulative_return'] = (1 + backtest_df['daily_return']).cumprod()
        
        # If the position is still open at the end, close it for metrics calculation
        if current_holdings_qty > 0 and len(trades) > 0 and trades[-1]['is_open']:
            last_date = backtest_df.index[-1]
            last_close = backtest_df.loc[last_date, 'close']
            
            gross_value = current_holdings_qty * last_close
            transaction_cost = gross_value * total_cost_rate
            net_value = gross_value - transaction_cost
            
            holding_days = (last_date - entry_date).days if entry_date else 0
            price_return = (last_close / entry_price - 1) if entry_price > 0 else 0
            trade_return = (net_value / trades[-1]['entry_value'] - 1) if trades[-1]['entry_value'] > 0 else 0
            
            trades[-1].update({
                'exit_date': last_date,
                'holding_days': holding_days,
                'exit_price': last_close,
                'price_return': price_return,
                'exit_value': gross_value,
                'exit_cost': transaction_cost,
                'trade_return': trade_return,
                'is_open': False  # Mark as closed for final analysis
            })
        
        # Compute performance metrics for strategy and buy & hold
        strategy_metrics = self._calculate_backtest_metrics(
            backtest_df, 
            trades, 
            'daily_return', 
            'strategy_cumulative_return', 
            initial_capital
        )
        
        buy_hold_metrics = self._calculate_backtest_metrics(
            backtest_df, 
            [], 
            'buy_hold_daily_return', 
            'buy_hold_cumulative_return', 
            initial_capital
        )
        
        # Create evaluation metrics comparison table
        metrics_comparison = pd.DataFrame({
            'Strategy': [
                strategy_metrics['total_return'],
                strategy_metrics['max_drawdown'],
                strategy_metrics['sharpe_ratio'],
                strategy_metrics['sortino_ratio'],
                strategy_metrics['calmar_ratio'],
                strategy_metrics['win_rate'],
                strategy_metrics['annualized_volatility'],
                strategy_metrics['annualized_return'],
                strategy_metrics.get('accuracy', np.nan),
                strategy_metrics.get('precision', np.nan),
                strategy_metrics.get('recall', np.nan),
                strategy_metrics.get('f1_score', np.nan)
            ],
            'Buy & Hold': [
                buy_hold_metrics['total_return'],
                buy_hold_metrics['max_drawdown'],
                buy_hold_metrics['sharpe_ratio'],
                buy_hold_metrics['sortino_ratio'],
                buy_hold_metrics['calmar_ratio'],
                buy_hold_metrics['win_rate'],
                buy_hold_metrics['annualized_volatility'],
                buy_hold_metrics['annualized_return'],
                buy_hold_metrics.get('accuracy', np.nan),
                buy_hold_metrics.get('precision', np.nan),
                buy_hold_metrics.get('recall', np.nan),
                buy_hold_metrics.get('f1_score', np.nan)
            ]
        }, index=[
            'Total Return',
            'Max Drawdown',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Win Rate',
            'Annualized Volatility',
            'Annualized Return',
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score'
        ])
        
        # Store the backtest results in the asset's data structure if possible
        if hasattr(self.data_loader, 'assets'):
            asset = self.data_loader.assets.get(asset_id)
            if asset and hasattr(asset, 'add_backtest_result'):
                # Create a descriptive backtest name based on the signals used
                signal_desc = []
                if usd_signals and len(usd_signals) > 0:
                    signal_desc.append(f"USD:{'+'.join(usd_signals)}")
                if btc_signals and len(btc_signals) > 0:
                    signal_desc.append(f"BTC:{'+'.join(btc_signals)}")
                
                backtest_name = "_".join(signal_desc)
                asset.add_backtest_result(backtest_name, {
                    'results': backtest_df,
                    'trades': trades,
                    'metrics': metrics_comparison,
                    'strategy_metrics': strategy_metrics,
                    'buy_hold_metrics': buy_hold_metrics,
                    'parameters': {
                        'initial_capital': initial_capital,
                        'trade_cost': trade_cost,
                        'slippage_pct': slippage_pct,
                        'usd_signals': usd_signals,
                        'btc_signals': btc_signals,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                })
                self.logger.info(f"Stored backtest results in asset {asset_id} with name '{backtest_name}'")
        elif hasattr(self.data_loader, 'get_asset_data'):
            asset = self.data_loader.get_asset_data(asset_id)
            if asset and hasattr(asset, 'add_backtest_result'):
                # Create a descriptive backtest name based on the signals used
                signal_desc = []
                if usd_signals and len(usd_signals) > 0:
                    signal_desc.append(f"USD:{'+'.join(usd_signals)}")
                if btc_signals and len(btc_signals) > 0:
                    signal_desc.append(f"BTC:{'+'.join(btc_signals)}")
                
                backtest_name = "_".join(signal_desc)
                asset.add_backtest_result(backtest_name, {
                    'results': backtest_df,
                    'trades': trades,
                    'metrics': metrics_comparison,
                    'strategy_metrics': strategy_metrics,
                    'buy_hold_metrics': buy_hold_metrics,
                    'parameters': {
                        'initial_capital': initial_capital,
                        'trade_cost': trade_cost,
                        'slippage_pct': slippage_pct,
                        'usd_signals': usd_signals,
                        'btc_signals': btc_signals,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                })
                self.logger.info(f"Stored backtest results in asset {asset_id} with name '{backtest_name}'")
        
        # Return the complete results
        return {
            'asset_id': asset_id,
            'results_df': backtest_df,
            'trades': trades,
            'metrics_comparison': metrics_comparison,
            'strategy_metrics': strategy_metrics,
            'buy_hold_metrics': buy_hold_metrics,
            'parameters': {
                'initial_capital': initial_capital,
                'trade_cost': trade_cost,
                'slippage_pct': slippage_pct,
                'usd_signals': usd_signals,
                'btc_signals': btc_signals,
                'start_date': start_date,
                'end_date': end_date
            }
        }
    
    def _calculate_backtest_metrics(self, backtest_df, trades, return_col, cumret_col, initial_capital):
        """
        Calculate detailed performance metrics for a backtest.
        
        Args:
            backtest_df (pd.DataFrame): DataFrame with backtest results
            trades (list): List of trade dictionaries
            return_col (str): Column name for daily returns
            cumret_col (str): Column name for cumulative returns
            initial_capital (float): Initial capital for the backtest
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Drop NaN values
        returns = backtest_df[return_col].dropna()
        
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'annualized_volatility': 0.0,
                'annualized_return': 0.0
            }
        
        # Calculate basic metrics
        total_return = backtest_df[cumret_col].iloc[-1] - 1.0 if not pd.isna(backtest_df[cumret_col].iloc[-1]) else 0.0
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(365) if daily_volatility else 0.0
        annual_return = (1 + total_return) ** (365 / len(returns)) - 1 if total_return > -1 else -1.0
        
        # Calculate drawdown
        cumulative = backtest_df[cumret_col].dropna()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
        
        # Risk metrics
        risk_free_rate = 0.0  # Assuming zero risk-free rate for simplicity
        excess_returns = returns - risk_free_rate
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(365) if len(negative_returns) > 0 else annual_volatility
        
        # Ratios
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility else 0.0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        # Calculate win rate from trades
        if trades:
            winning_trades = [t for t in trades if t.get('trade_return', 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
        else:
            win_rate = 0.0
            
        # Signal accuracy metrics
        if 'final_decision_shifted' in backtest_df.columns and return_col in backtest_df.columns:
            # Only include rows where we have both signal and return data
            mask = backtest_df['final_decision_shifted'].notna() & backtest_df[return_col].notna()
            valid_data = backtest_df.loc[mask]
            
            if len(valid_data) > 0:
                # True positive: Signal = 1 and return > 0
                tp = ((valid_data['final_decision_shifted'] == 1) & (valid_data[return_col] > 0)).sum()
                
                # False positive: Signal = 1 but return <= 0
                fp = ((valid_data['final_decision_shifted'] == 1) & (valid_data[return_col] <= 0)).sum()
                
                # True negative: Signal = 0 and return <= 0
                tn = ((valid_data['final_decision_shifted'] == 0) & (valid_data[return_col] <= 0)).sum()
                
                # False negative: Signal = 0 but return > 0
                fn = ((valid_data['final_decision_shifted'] == 0) & (valid_data[return_col] > 0)).sum()
                
                # Calculate metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                accuracy = precision = recall = f1_score = 0.0
        else:
            accuracy = precision = recall = f1_score = 0.0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'annualized_volatility': annual_volatility,
            'annualized_return': annual_return,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def get_backtest_summary_table(self, results):
        """
        Format the backtest results into a nicely formatted summary table.
        
        Args:
            results: Dictionary with backtest results from backtest_assets
            
        Returns:
            pd.DataFrame: Formatted summary table
        """
        # Check if results contains a summary
        if 'summary' not in results or results['summary'].empty:
            self.logger.warning("No summary data in backtest results")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        summary_df = results['summary'].copy()
        
        # Create a formatted summary table with the required columns
        formatted_df = pd.DataFrame(index=summary_df.index)
        
        # Required columns in the desired order from the image
        formatted_df['Total Return (%)'] = summary_df['Total Return (%)'].round(2)
        formatted_df['Peak Return (%)'] = summary_df['Peak Return (%)'].round(2)
        formatted_df['Buy & Hold Return (%)'] = summary_df['Buy & Hold Return (%)'].round(2)
        formatted_df['Buy & Hold Peak (%)'] = summary_df['Buy & Hold Peak (%)'].round(2)
        formatted_df['Sharpe'] = summary_df['Sharpe Ratio'].round(2)
        formatted_df['Buy & Hold Sharpe'] = summary_df['Buy & Hold Sharpe'].round(2)
        formatted_df['Sortino Ratio'] = summary_df['Sortino Ratio'].round(2)
        formatted_df['Buy & Hold Sortino'] = summary_df['Buy & Hold Sortino'].round(2)
        formatted_df['Max Drawdown (%)'] = summary_df['Max Drawdown (%)'].round(2)
        formatted_df['Buy & Hold Max Drawdown (%)'] = summary_df['Buy & Hold Drawdown (%)'].round(2)
        formatted_df['Annualized Volatility (%)'] = summary_df['Annualized Volatility (%)'].round(2)
        formatted_df['Buy & Hold Volatility (%)'] = summary_df['Buy & Hold Volatility (%)'].round(2)
        formatted_df['Trade Count'] = summary_df['Trade Count']
        formatted_df['Total Cost'] = summary_df['Total Cost'].round(2)
        
        return formatted_df

    def backtest_assets(self,
                      asset_ids,
                      start_date,
                      end_date,
                      initial_capital=10000.0,
                      btc_cost=0.001,
                      alt_cost=0.005,
                      slippage_pct=0.0,
                      usd_signals=None,
                      btc_signals=None,
                      use_btc_filter=False):
        """Run backtests for multiple assets and generate a summary table.
        
        Args:
            asset_ids: List of asset IDs to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for each asset's backtest
            btc_cost: Transaction cost for Bitcoin
            alt_cost: Transaction cost for altcoins
            slippage_pct: Slippage percentage
            usd_signals: USD signals to use
            btc_signals: BTC signals to use
            use_btc_filter: Whether to use Bitcoin filter for altcoins
            
        Returns:
            Dictionary with asset results and summary table
        """
        # Validate inputs
        if not asset_ids:
            self.logger.error("No asset IDs provided")
            return None
            
        # Convert asset_ids to list if it's a single string
        if isinstance(asset_ids, str):
            asset_ids = [asset_ids]
            
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Log what we're doing
        self.logger.info(f"Running backtests for {len(asset_ids)} assets")
        
        # Run a backtest for each asset
        asset_results = {}
        summary_data = []
        
        # Create a data loader to get tickers
        if hasattr(self.data_loader, 'get_ticker_from_id'):
            ticker_mapping = lambda asset_id: self.data_loader.get_ticker_from_id(asset_id)
        else:
            # Fallback mapping for common assets
            ticker_mapping = lambda asset_id: {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'solana': 'SOL',
                'dogecoin': 'DOGE',
                'chainlink': 'LINK',
                'pendle': 'PENDLE'
            }.get(asset_id, asset_id.upper()[:3])
        
        # Set up Bitcoin filter if requested
        bitcoin_decisions = None
        if use_btc_filter and 'bitcoin' in asset_ids:
            self.logger.info(f"Using Bitcoin as a filter with signals: {usd_signals}")
            # First run backtest on Bitcoin
            btc_result = self.backtest_asset(
                asset_id='bitcoin',
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                trade_cost=btc_cost,
                slippage_pct=slippage_pct,
                usd_signals=usd_signals,
                btc_signals=None  # No BTC signals for Bitcoin itself
            )
            
            if btc_result and 'results_df' in btc_result:
                # Extract Bitcoin decisions from the results
                bitcoin_decisions = btc_result['results_df']['final_decision_shifted']
                self.logger.info(f"Generated Bitcoin decisions for filtering: {len(bitcoin_decisions)} entries")
                
                # Store Bitcoin result
                asset_results['bitcoin'] = btc_result
                
                # Extract ticker for summary
                ticker = ticker_mapping('bitcoin')
                
                # Extract metrics directly from result or from metrics_comparison
                metrics = btc_result['metrics_comparison']
                strategy_metrics = btc_result.get('strategy_metrics', {})
                buy_hold_metrics = btc_result.get('buy_hold_metrics', {})
                
                # If metrics are missing, extract from the metrics_comparison DataFrame
                if not strategy_metrics or not buy_hold_metrics:
                    try:
                        strategy_metrics = {
                            'total_return': metrics.loc['Total Return', 'Strategy'],
                            'max_drawdown': metrics.loc['Max Drawdown', 'Strategy'],
                            'sharpe_ratio': metrics.loc['Sharpe Ratio', 'Strategy'],
                            'sortino_ratio': metrics.loc['Sortino Ratio', 'Strategy'],
                            'annualized_volatility': metrics.loc['Annualized Volatility', 'Strategy'],
                            'annualized_return': metrics.loc['Annualized Return', 'Strategy']
                        }
                        
                        buy_hold_metrics = {
                            'total_return': metrics.loc['Total Return', 'Buy & Hold'],
                            'max_drawdown': metrics.loc['Max Drawdown', 'Buy & Hold'],
                            'sharpe_ratio': metrics.loc['Sharpe Ratio', 'Buy & Hold'],
                            'sortino_ratio': metrics.loc['Sortino Ratio', 'Buy & Hold'],
                            'annualized_volatility': metrics.loc['Annualized Volatility', 'Buy & Hold'],
                            'annualized_return': metrics.loc['Annualized Return', 'Buy & Hold']
                        }
                        self.logger.info(f"Successfully extracted metrics from metrics_comparison for bitcoin")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract metrics from metrics_comparison for bitcoin: {str(e)}")
                
                # Calculate peak performance (maximum portfolio value)
                peak_return = 0.0
                if 'portfolio_value' in btc_result['results_df'].columns:
                    max_portfolio_value = btc_result['results_df']['portfolio_value'].max()
                    peak_return = (max_portfolio_value / initial_capital) - 1
                
                # Calculate buy & hold peak return
                bh_peak_return = 0.0
                if 'buy_hold_value' in btc_result['results_df'].columns:
                    max_bh_value = btc_result['results_df']['buy_hold_value'].max()
                    bh_peak_return = (max_bh_value / initial_capital) - 1
                
                # Calculate total costs
                total_cost = 0.0
                if 'trades' in btc_result:
                    for trade in btc_result['trades']:
                        entry_cost = abs(trade.get('entry_cost', 0.0))
                        exit_cost = abs(trade.get('exit_cost', 0.0)) if trade.get('exit_cost') is not None else 0.0
                        total_cost += entry_cost + exit_cost
                
                # Add to summary data
                summary_data.append({
                    'Asset': ticker,
                    'Total Return (%)': strategy_metrics['total_return'] * 100,
                    'Peak Return (%)': peak_return * 100,
                    'Buy & Hold Return (%)': buy_hold_metrics['total_return'] * 100,
                    'Buy & Hold Peak (%)': bh_peak_return * 100,
                    'Sharpe Ratio': strategy_metrics['sharpe_ratio'],
                    'Buy & Hold Sharpe': buy_hold_metrics['sharpe_ratio'],
                    'Sortino Ratio': strategy_metrics['sortino_ratio'],
                    'Buy & Hold Sortino': buy_hold_metrics['sortino_ratio'],
                    'Max Drawdown (%)': strategy_metrics['max_drawdown'] * 100,
                    'Buy & Hold Drawdown (%)': buy_hold_metrics['max_drawdown'] * 100,
                    'Annualized Volatility (%)': strategy_metrics['annualized_volatility'] * 100,
                    'Buy & Hold Volatility (%)': buy_hold_metrics['annualized_volatility'] * 100,
                    'Trade Count': len(btc_result['trades']),
                    'Total Cost': total_cost
                })
            else:
                self.logger.warning("Bitcoin filter enabled but no decisions generated")
                # Continue without filter
                use_btc_filter = False
        
        # Process other assets
        for asset_id in asset_ids:
            # Skip Bitcoin if already processed
            if asset_id == 'bitcoin' and bitcoin_decisions is not None:
                continue
                
            # Determine transaction cost and ticker
            if asset_id == 'bitcoin':
                trade_cost = btc_cost
            else:
                trade_cost = alt_cost
                
            # Get ticker for this asset
            ticker = ticker_mapping(asset_id)
            
            self.logger.info(f"Running backtest for {asset_id} ({ticker})")
            
            # Run backtest
            result = None
            try:
                # For altcoins with Bitcoin filter
                if use_btc_filter and asset_id != 'bitcoin' and bitcoin_decisions is not None:
                    # Apply Bitcoin filter - this is a simplified approach
                    # A real implementation would need to be more sophisticated
                    
                    # First run standard backtest
                    result = self.backtest_asset(
                        asset_id=asset_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        trade_cost=trade_cost,
                        slippage_pct=slippage_pct,
                        usd_signals=usd_signals,
                        btc_signals=btc_signals
                    )
                    
                    if result and 'results_df' in result:
                        # Apply Bitcoin filter - only buy when Bitcoin signal is also buy
                        df = result['results_df'].copy()
                        # Align Bitcoin decisions to match this asset's dates
                        btc_aligned = bitcoin_decisions.reindex(df.index, fill_value=0)
                        
                        # Store original decision for reference
                        df['original_decision'] = df['final_decision_shifted']
                        
                        # Combine: only buy when both asset and Bitcoin signals are positive
                        df['final_decision_shifted'] = (df['final_decision_shifted'].astype(bool) & 
                                                       btc_aligned.astype(bool)).astype(int)
                        
                        # TODO: Re-run trading simulation with new signals
                        # This would need a more complex implementation
                        
                        self.logger.info(f"Applied Bitcoin filter to {asset_id}: "
                                        f"Original buy signals: {df['original_decision'].sum()}, "
                                        f"After filter: {df['final_decision_shifted'].sum()}")
                else:
                    # Standard backtest without filter
                    result = self.backtest_asset(
                        asset_id=asset_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        trade_cost=trade_cost,
                        slippage_pct=slippage_pct,
                        usd_signals=usd_signals,
                        btc_signals=btc_signals
                    )
            except Exception as e:
                self.logger.error(f"Error running backtest for {asset_id}: {str(e)}")
                continue
            
            # Store result and create summary entry
            if result and 'metrics_comparison' in result:
                asset_results[asset_id] = result
                metrics = result['metrics_comparison']
                
                # Extract metrics directly from result or from metrics_comparison
                strategy_metrics = result.get('strategy_metrics', {})
                buy_hold_metrics = result.get('buy_hold_metrics', {})
                
                # If metrics are missing, extract from the metrics_comparison DataFrame
                if not strategy_metrics or not buy_hold_metrics:
                    try:
                        strategy_metrics = {
                            'total_return': metrics.loc['Total Return', 'Strategy'],
                            'max_drawdown': metrics.loc['Max Drawdown', 'Strategy'],
                            'sharpe_ratio': metrics.loc['Sharpe Ratio', 'Strategy'],
                            'sortino_ratio': metrics.loc['Sortino Ratio', 'Strategy'],
                            'annualized_volatility': metrics.loc['Annualized Volatility', 'Strategy'],
                            'annualized_return': metrics.loc['Annualized Return', 'Strategy']
                        }
                        
                        buy_hold_metrics = {
                            'total_return': metrics.loc['Total Return', 'Buy & Hold'],
                            'max_drawdown': metrics.loc['Max Drawdown', 'Buy & Hold'],
                            'sharpe_ratio': metrics.loc['Sharpe Ratio', 'Buy & Hold'],
                            'sortino_ratio': metrics.loc['Sortino Ratio', 'Buy & Hold'],
                            'annualized_volatility': metrics.loc['Annualized Volatility', 'Buy & Hold'],
                            'annualized_return': metrics.loc['Annualized Return', 'Buy & Hold']
                        }
                        self.logger.info(f"Successfully extracted metrics from metrics_comparison for {asset_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract metrics from metrics_comparison: {str(e)}")
                
                # Calculate peak performance (maximum portfolio value)
                peak_return = 0.0
                if 'portfolio_value' in result['results_df'].columns:
                    max_portfolio_value = result['results_df']['portfolio_value'].max()
                    peak_return = (max_portfolio_value / initial_capital) - 1
                
                # Calculate buy & hold peak return
                bh_peak_return = 0.0
                if 'buy_hold_value' in result['results_df'].columns:
                    max_bh_value = result['results_df']['buy_hold_value'].max()
                    bh_peak_return = (max_bh_value / initial_capital) - 1
                
                # Calculate total costs
                total_cost = 0.0
                if 'trades' in result:
                    for trade in result['trades']:
                        entry_cost = abs(trade.get('entry_cost', 0.0))
                        exit_cost = abs(trade.get('exit_cost', 0.0)) if trade.get('exit_cost') is not None else 0.0
                        total_cost += entry_cost + exit_cost
                
                # Add to summary data
                summary_data.append({
                    'Asset': ticker,
                    'Total Return (%)': strategy_metrics['total_return'] * 100,
                    'Peak Return (%)': peak_return * 100,
                    'Buy & Hold Return (%)': buy_hold_metrics['total_return'] * 100,
                    'Buy & Hold Peak (%)': bh_peak_return * 100,
                    'Sharpe Ratio': strategy_metrics['sharpe_ratio'],
                    'Buy & Hold Sharpe': buy_hold_metrics['sharpe_ratio'],
                    'Sortino Ratio': strategy_metrics['sortino_ratio'],
                    'Buy & Hold Sortino': buy_hold_metrics['sortino_ratio'],
                    'Max Drawdown (%)': strategy_metrics['max_drawdown'] * 100,
                    'Buy & Hold Drawdown (%)': buy_hold_metrics['max_drawdown'] * 100,
                    'Annualized Volatility (%)': strategy_metrics['annualized_volatility'] * 100,
                    'Buy & Hold Volatility (%)': buy_hold_metrics['annualized_volatility'] * 100,
                    'Trade Count': len(result['trades']),
                    'Total Cost': total_cost
                })
            else:
                self.logger.warning(f"No valid results for {asset_id}")
        
        # Create summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            # Set Asset column as index for better display
            if 'Asset' in summary_df.columns:
                summary_df.set_index('Asset', inplace=True)
            
            # Sort by total return descending
            if 'Total Return (%)' in summary_df.columns:
                summary_df = summary_df.sort_values('Total Return (%)', ascending=False)
        else:
            summary_df = pd.DataFrame()
        
        # Create the dictionary with results
        results_dict = {
            'asset_results': asset_results,
            'summary': summary_df,
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'btc_cost': btc_cost,
                'alt_cost': alt_cost,
                'slippage_pct': slippage_pct,
                'usd_signals': usd_signals,
                'btc_signals': btc_signals,
                'use_btc_filter': use_btc_filter
            }
        }
        
        # Format and add the summary table
        formatted_summary = self.get_backtest_summary_table(results_dict)
        results_dict['summary'] = formatted_summary
        
        return results_dict
