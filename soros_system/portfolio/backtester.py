"""
Simple portfolio backtester for the Soros System.

This module implements a portfolio backtester that uses binary signal logic
for trading decisions.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from ..analysis.metrics import MetricsCalculator


class Backtester:
    """
    Backtest portfolio strategies using binary signal logic.
    
    This class simulates trades based on binary decisions (1 for buy, 0 for sell).
    """
    
    def __init__(
        self, 
        initial_capital: float = 10000.0,
        trade_cost: float = 0.001,
        slippage_pct: float = 0.0,
        portfolio_manager=None, 
        data_loader=None, 
        analyzer=None
    ):
        """Initialize the backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
            trade_cost: Trading cost as a fraction
            slippage_pct: Slippage percentage for trading
            portfolio_manager: Manager for portfolios
            data_loader: Loader for asset data
            analyzer: PortfolioAnalyzer instance
        """
        self.logger = logging.getLogger(__name__)
        self.portfolio_manager = portfolio_manager
        self.data_loader = data_loader
        self.analyzer = analyzer
        self.metrics_calculator = MetricsCalculator()
        self.initial_capital = initial_capital
        self.trade_cost = trade_cost
        self.slippage_pct = slippage_pct
        self.debug = False
        
        # If analyzer is provided, use it for data loading
        if analyzer is not None:
            self.data_loader = analyzer
            self.logger.info("Using PortfolioAnalyzer for data loading")
    
    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        decisions: Dict[str, pd.Series],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        asset_specific_costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run a backtest based on price data and binary decisions.
        
        Args:
            price_data: Dictionary mapping asset IDs to price data
            decisions: Dictionary mapping asset IDs to decision series (1 for buy, 0 for sell)
            start_date: Start date for the backtest
            end_date: End date for the backtest
            asset_specific_costs: Dictionary mapping asset IDs to trading costs.
                                 Use 'default' as key for the default cost.
            
        Returns:
            dict: Dictionary with backtest results
        """
        self.logger.info("Running backtest...")
        
        # Determine date range if not provided
        if start_date is None or end_date is None:
            all_dates = set()
            for asset_id, prices in price_data.items():
                all_dates.update(prices.index)
            date_range = sorted(all_dates)
            
            if start_date is None:
                start_date = date_range[0] if date_range else None
            if end_date is None:
                end_date = date_range[-1] if date_range else None
                
        if start_date is None or end_date is None:
            self.logger.error("Could not determine date range for backtest")
            return {'success': False, 'error': 'No date range'}
            
        self.logger.info(f"Backtest date range: {start_date} to {end_date}")
        
        # Initialize portfolio tracking
        portfolio_value = pd.Series(self.initial_capital, index=pd.date_range(start_date, end_date))
        cash = self.initial_capital
        positions = {asset_id: 0.0 for asset_id in price_data.keys()}
        trades = []
        
        # Run backtest day by day
        for day in pd.date_range(start_date, end_date):
            # Calculate current asset values
            current_asset_values = {}
            for asset_id, prices in price_data.items():
                # Skip if no price data for this day
                if day not in prices.index:
                    continue
                    
                # Get current price
                current_price = prices.loc[day, 'close']
                
                # Calculate current value
                current_value = positions[asset_id] * current_price
                current_asset_values[asset_id] = current_value
            
            # Calculate total portfolio value (cash + assets)
            portfolio_value[day] = cash + sum(current_asset_values.values())
            
            # Check for trading signals and execute trades
            for asset_id in price_data.keys():
                # Skip if no decision for this day or asset
                if asset_id not in decisions or day not in decisions[asset_id].index:
                    continue
                    
                # Get decision (1 = buy, 0 = sell)
                decision = decisions[asset_id].loc[day]
                
                # Get current position and price
                current_position = positions[asset_id]
                if day not in price_data[asset_id].index:
                    continue
                current_price = price_data[asset_id].loc[day, 'close']
                
                # Determine trading cost for this asset
                if asset_specific_costs is not None:
                    # Use asset-specific cost if provided, otherwise use default
                    if asset_id in asset_specific_costs:
                        trade_cost = asset_specific_costs[asset_id]
                    elif 'default' in asset_specific_costs:
                        trade_cost = asset_specific_costs['default']
                    else:
                        trade_cost = self.trade_cost
                else:
                    trade_cost = self.trade_cost
                
                # Log the trading cost
                if self.debug:
                    self.logger.debug(f"Using trade cost {trade_cost:.3%} for {asset_id}")
                
                # Execute trades based on decision
                if decision == 1 and current_position == 0:
                    # Buy signal
                    # Allocate a portion of the portfolio
                    allocation = 0.25  # 25% of portfolio
                    trade_value = portfolio_value[day] * allocation
                    
                    # Calculate trade size (including costs)
                    trade_size = trade_value / (current_price * (1 + self.slippage_pct))
                    trade_cost_amount = trade_value * trade_cost
                    
                    # Execute trade
                    if cash >= trade_value + trade_cost_amount:
                        # Update cash and position
                        cash -= (trade_value + trade_cost_amount)
                        positions[asset_id] = trade_size
                        
                        # Log trade
                        trades.append({
                            'date': day,
                            'asset_id': asset_id,
                            'type': 'buy',
                            'price': current_price,
                            'amount': trade_size,
                            'value': trade_value,
                            'cost': trade_cost_amount,
                            'cost_rate': trade_cost
                        })
                        
                        self.logger.debug(
                            f"Buy {asset_id} on {day}: {trade_size} units at {current_price}, "
                            f"value: {trade_value}, cost: {trade_cost_amount} ({trade_cost:.3%})"
                        )
                        
                elif decision == 0 and current_position > 0:
                    # Sell signal
                    trade_value = current_position * current_price * (1 - self.slippage_pct)
                    trade_cost_amount = trade_value * trade_cost
                    
                    # Execute trade
                    cash += (trade_value - trade_cost_amount)
                    positions[asset_id] = 0.0
                    
                    # Log trade
                    trades.append({
                        'date': day,
                        'asset_id': asset_id,
                        'type': 'sell',
                        'price': current_price,
                        'amount': current_position,
                        'value': trade_value,
                        'cost': trade_cost_amount,
                        'cost_rate': trade_cost
                    })
                    
                    self.logger.debug(
                        f"Sell {asset_id} on {day}: {current_position} units at {current_price}, "
                        f"value: {trade_value}, cost: {trade_cost_amount} ({trade_cost:.3%})"
                    )
        
        # Calculate final portfolio metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate metrics
        if len(portfolio_value) > 1:
            start_value = portfolio_value.iloc[0]
            end_value = portfolio_value.iloc[-1]
            total_return = (end_value / start_value) - 1
            
            # Daily returns
            daily_returns = portfolio_value.pct_change().dropna()
            
            # Annualized return
            days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
            if days > 0:
                years = days / 365.0
                annualized_return = ((1 + total_return) ** (1 / years)) - 1
            else:
                annualized_return = 0
                
            # Volatility
            daily_volatility = daily_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Sharpe Ratio (using 0 as risk-free rate for simplicity)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Win Rate (if trades were made)
            if not trades_df.empty:
                buy_trades = trades_df[trades_df['type'] == 'buy']
                sell_trades = trades_df[trades_df['type'] == 'sell']
                
                # Match buys with sells (simplified)
                if len(buy_trades) > 0 and len(sell_trades) > 0:
                    # Simple approach - pair buys and sells by asset
                    profitable_trades = 0
                    total_paired_trades = 0
                    
                    for asset_id in trades_df['asset_id'].unique():
                        asset_buys = buy_trades[buy_trades['asset_id'] == asset_id]
                        asset_sells = sell_trades[sell_trades['asset_id'] == asset_id]
                        
                        for _, buy in asset_buys.iterrows():
                            # Find the corresponding sell (simplified)
                            asset_sells_after_buy = asset_sells[asset_sells['date'] > buy['date']]
                            if not asset_sells_after_buy.empty:
                                sell = asset_sells_after_buy.iloc[0]
                                
                                # Calculate profit
                                buy_value = buy['value']
                                sell_value = sell['value']
                                profit = sell_value - buy_value
                                
                                if profit > 0:
                                    profitable_trades += 1
                                    
                                total_paired_trades += 1
                    
                    win_rate = profitable_trades / total_paired_trades if total_paired_trades > 0 else 0
                else:
                    win_rate = 0
            else:
                win_rate = 0
                
            results = {
                'portfolio_value': portfolio_value,
                'trades': trades_df,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'success': True
            }
        else:
            results = {
                'portfolio_value': portfolio_value,
                'trades': trades_df,
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'success': True
            }
            
        return results 