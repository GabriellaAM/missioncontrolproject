#!/usr/bin/env python

import argparse
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_mlflow, create_experiment


class StrategyRunner:
    
    def __init__(self, strategy_name: str, asset_name: str):
        self.strategy_name = strategy_name
        self.asset_name = asset_name
        self.strategy = self._load_strategy()
        self.asset_config = self._load_asset_config()
        self._setup_mlflow()
    
    def _load_strategy(self):
        pass
    
    def _load_asset_config(self):
        pass
    
    def _setup_mlflow(self):
        setup_mlflow()
        create_experiment(self.strategy_name)
    
    def run(self, start_date: str, end_date: str, train_test_split: float = 0.75):
        pass


def main():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--strategy', required=True, help='Strategy name')
    parser.add_argument('--asset', required=True, help='Asset name')
    parser.add_argument('--start-date', required=True, help='Start date')
    parser.add_argument('--end-date', required=True, help='End date')
    parser.add_argument('--train-test-split', type=float, default=0.75, help='Train/test split')
    
    args = parser.parse_args()
    
    runner = StrategyRunner(args.strategy, args.asset)
    runner.run(args.start_date, args.end_date, args.train_test_split)


if __name__ == "__main__":
    main()