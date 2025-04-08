#!/usr/bin/env python
"""
Example script demonstrating the unified PortfolioAnalyzer interface.

This script shows how to use the new PortfolioAnalyzer class to:
1. Load asset data for a subset of assets
2. Evaluate signals for their effectiveness
3. Track signal activation events
4. Run a backtest using the event-driven logic
5. Get current recommendations
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pprint import pprint

# Import the unified interface
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate the PortfolioAnalyzer."""
    # Create analyzer
    analyzer = PortfolioAnalyzer(
        data_dir='data',
        signal_threshold=0.0,
        use_meta_labeling=False,
        lookback_days=365,
        parallelize=True
    )
    
    # Using a small subset for the example
    assets_subset = ['bitcoin', 'ethereum', 'solana', 'polkadot']
    
    # Set date range for analysis
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    # Step 1: Load data
    logger.info("Step 1: Loading data for a subset of assets")
    loaded_assets = analyzer.load_data(
        assets=assets_subset,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print loaded assets
    logger.info(f"Loaded {len(loaded_assets)} assets: {', '.join(loaded_assets.keys())}")
    
    # Step 2: Register signals
    logger.info("\nStep 2: Registering signals for Bitcoin")
    # We'll use a subset of signals for the example
    btc_signals = analyzer.register_signals(
        asset_id='bitcoin',
        signal_names=['rsi_signal', 'trend_signal', 'volatility_signal']
    )
    
    logger.info(f"Registered {len(btc_signals)} signals for Bitcoin")
    
    # Step 3: Evaluate signals
    logger.info("\nStep 3: Evaluating signals for effectiveness")
    evaluation_results = analyzer.evaluate_signals(
        asset_ids=['bitcoin'],
        force_recalculate=True
    )
    
    # Print signal evaluation summary
    bitcoin_results = evaluation_results.get('bitcoin', {})
    if bitcoin_results:
        logger.info("Signal evaluation results for Bitcoin:")
        for signal_name, result in bitcoin_results.items():
            effective = result.get('overall_effectiveness', False)
            weight = result.get('weight', 0.0)
            holding_period = result.get('optimal_holding_period', 0)
            logger.info(
                f"  {signal_name}: Effective={effective}, Weight={weight:.4f}, "
                f"Holding Period={holding_period} days"
            )
    
    # Step 4: Get effective signals
    logger.info("\nStep 4: Getting effective signals for Bitcoin")
    effective_signals = analyzer.get_effective_signals('bitcoin')
    
    logger.info(f"Found {len(effective_signals)} effective signals for Bitcoin:")
    for name, signal in effective_signals.items():
        logger.info(
            f"  {name}: Weight={signal.weight:.4f}, Period={signal.optimal_holding_period} days"
        )
    
    # Step 5: Track signal events
    logger.info("\nStep 5: Tracking signal activation events")
    activation_dates = analyzer.track_signal_events(
        asset_id='bitcoin',
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        use_effective_signals_only=True
    )
    
    # Print activation dates
    total_activations = sum(len(dates) for dates in activation_dates.values())
    logger.info(f"Found {total_activations} activations across {len(activation_dates)} signals")
    
    for signal_name, dates in activation_dates.items():
        if dates:
            logger.info(f"  {signal_name}: {len(dates)} activations")
            # Print first few activations
            if len(dates) > 0:
                date_strs = [d.strftime('%Y-%m-%d') for d in dates[:3]]
                logger.info(f"    First activations: {', '.join(date_strs)}")
    
    # Step 6: Run a backtest
    logger.info("\nStep 6: Running an event-driven backtest")
    backtest_results = analyzer.run_backtest(
        backtest_name="example_backtest",
        assets=['bitcoin'],
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        trade_cost=0.001,
        slippage_pct=0.0,
        mode='event_driven'
    )
    
    # Print backtest results
    if backtest_results:
        logger.info("Backtest results summary:")
        
        # Print metrics for Bitcoin
        btc_metrics = backtest_results.get('metrics', {}).get('bitcoin', {})
        if btc_metrics:
            strategy_metrics = btc_metrics.get('Strategy', {})
            benchmark_metrics = btc_metrics.get('Benchmark', {})
            
            logger.info("Bitcoin performance metrics:")
            logger.info(f"  Total Return: {strategy_metrics.get('Total Return (%)', 0.0):.2f}%")
            logger.info(f"  Benchmark Return: {benchmark_metrics.get('Total Return (%)', 0.0):.2f}%")
            logger.info(f"  Sharpe Ratio: {strategy_metrics.get('Sharpe Ratio', 0.0):.2f}")
            logger.info(f"  Max Drawdown: {strategy_metrics.get('Max Drawdown (%)', 0.0):.2f}%")
            logger.info(f"  Total Trades: {strategy_metrics.get('Total Trades', 0)}")
    
    # Step 7: Get current recommendations
    logger.info("\nStep 7: Getting current recommendations")
    # Use the last date in our data range as "current"
    current_date = pd.to_datetime(end_date)
    
    recommendations = analyzer.get_current_recommendations(
        asset_ids=['bitcoin', 'ethereum'],
        current_date=current_date
    )
    
    # Print recommendations
    logger.info(f"Recommendations for {current_date.strftime('%Y-%m-%d')}:")
    for asset_id, rec in recommendations.items():
        decision = "EXPOSED" if rec['decision'] == 1 else "CASH"
        logger.info(
            f"  {asset_id}: {decision} (Weight: {rec['combined_weight']:.4f}, "
            f"Active Signals: {rec['active_signals_count']})"
        )
    
    logger.info("\nExample complete!")

if __name__ == "__main__":
    main() 