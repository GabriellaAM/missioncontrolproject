#!/usr/bin/env python
"""
Example script demonstrating signal persistence functionality.

This script shows how to:
1. Use the SignalEvaluationStorage to persist signal evaluations
2. Load pre-evaluated signals instead of recalculating them
3. Explicitly re-evaluate signals when needed
4. View and manage the signal evaluation storage
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
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
    """Main function demonstrating signal persistence."""
    # Define our test assets
    test_assets = ['bitcoin', 'ethereum', 'solana']
    
    # Step 1: Create analyzer with persistence enabled (default)
    logger.info("Step 1: Creating analyzer with persistence enabled")
    analyzer = PortfolioAnalyzer(
        data_dir='data',
        signal_threshold=0.0,
        use_stored_evaluations=True,  # Enable signal persistence (default)
        lookback_days=365,
        signal_storage_file='demo_signal_evaluations.json',  # Custom file for demo
        verbose=True
    )
    
    # Load assets
    logger.info("\nLoading test assets")
    analyzer.load_data(assets=test_assets)
    
    # Step 2: Check if we have any stored evaluations
    logger.info("\nStep 2: Checking signal storage status")
    storage_info = analyzer.get_signal_storage_info()
    pprint(storage_info)
    
    # Step 3: Evaluate signals for the first time
    # This will calculate evaluations and store them
    logger.info("\nStep 3: Evaluating signals for the first time")
    t_start = datetime.now()
    evaluation_results = analyzer.evaluate_signals(asset_ids=test_assets)
    t_end = datetime.now()
    
    # Calculate and display time taken
    time_taken = (t_end - t_start).total_seconds()
    logger.info(f"First evaluation took {time_taken:.2f} seconds")
    
    # Display sample of results
    if evaluation_results and 'bitcoin' in evaluation_results:
        bitcoin_results = evaluation_results['bitcoin']
        logger.info(f"Evaluated {len(bitcoin_results)} signals for bitcoin")
        
        # Show sample evaluation
        for signal_name, result in list(bitcoin_results.items())[:2]:  # First two signals
            effective = result.get('overall_effectiveness', False)
            weight = result.get('weight', 0.0)
            logger.info(f"  {signal_name}: Effective={effective}, Weight={weight:.4f}")
    
    # Step 4: Check storage info after first evaluation
    logger.info("\nStep 4: Checking storage after first evaluation")
    storage_info = analyzer.get_signal_storage_info()
    pprint(storage_info)
    
    # Step 5: Evaluate signals again - should use stored evaluations
    logger.info("\nStep 5: Evaluating signals again (should use stored evaluations)")
    t_start = datetime.now()
    second_evaluation = analyzer.evaluate_signals(asset_ids=test_assets)
    t_end = datetime.now()
    
    # Calculate and display time taken
    time_taken = (t_end - t_start).total_seconds()
    logger.info(f"Second evaluation took {time_taken:.2f} seconds (should be much faster)")
    
    # Show effective signals for bitcoin from storage
    logger.info("\nGetting effective signals for bitcoin from storage")
    effective_signals = analyzer.get_effective_signals('bitcoin')
    logger.info(f"Found {len(effective_signals)} effective signals for bitcoin")
    
    for name, signal in list(effective_signals.items())[:3]:  # First three signals
        logger.info(f"  {name}: Weight={signal.weight:.4f}, Period={signal.optimal_holding_period}")
    
    # Step 6: Force re-evaluation of signals for one asset
    logger.info("\nStep 6: Forcing re-evaluation of signals for ethereum")
    t_start = datetime.now()
    ethereum_reevaluation = analyzer.force_reevaluate_signals(asset_ids=['ethereum'])
    t_end = datetime.now()
    
    # Calculate and display time taken
    time_taken = (t_end - t_start).total_seconds()
    logger.info(f"Forced re-evaluation took {time_taken:.2f} seconds")
    
    if ethereum_reevaluation and 'ethereum' in ethereum_reevaluation:
        logger.info(f"Re-evaluated {len(ethereum_reevaluation['ethereum'])} signals for ethereum")
    
    # Step 7: Create a new analyzer with persistence disabled
    logger.info("\nStep 7: Creating a new analyzer with persistence disabled")
    no_storage_analyzer = PortfolioAnalyzer(
        data_dir='data',
        use_stored_evaluations=False,  # Disable signal persistence
        lookback_days=365
    )
    
    # Load bitcoin
    no_storage_analyzer.load_data(assets=['bitcoin'])
    
    # Evaluate signals without persistence
    logger.info("Evaluating bitcoin signals without persistence")
    t_start = datetime.now()
    no_storage_results = no_storage_analyzer.evaluate_signals(asset_ids=['bitcoin'])
    t_end = datetime.now()
    
    # Calculate and display time taken
    time_taken = (t_end - t_start).total_seconds()
    logger.info(f"Evaluation without persistence took {time_taken:.2f} seconds")
    
    # Step 8: Demonstrate using evaluations in a backtest
    logger.info("\nStep 8: Using stored evaluations in a backtest")
    # Backtest bitcoin using stored signal evaluations
    backtest_results = analyzer.run_backtest(
        backtest_name="persistence_demo",
        assets=['bitcoin'],
        start_date='2022-01-01',
        end_date='2023-01-01',
        initial_capital=10000.0
    )
    
    # Print backtest results
    if backtest_results:
        logger.info("Bitcoin backtest results summary:")
        btc_metrics = backtest_results.get('metrics', {}).get('bitcoin', {}).get('Strategy', {})
        if btc_metrics:
            logger.info(f"  Total Return: {btc_metrics.get('Total Return (%)', 0.0):.2f}%")
            logger.info(f"  Sharpe Ratio: {btc_metrics.get('Sharpe Ratio', 0.0):.2f}")
            logger.info(f"  Max Drawdown: {btc_metrics.get('Max Drawdown (%)', 0.0):.2f}%")
            logger.info(f"  Total Trades: {btc_metrics.get('Total Trades', 0)}")
    
    logger.info("\nSignal persistence example complete!")


if __name__ == "__main__":
    main() 