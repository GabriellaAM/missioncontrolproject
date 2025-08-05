"""
Enhanced SOROS interface with meta-labeling and zenML integration.

This module provides the modern interface for the SOROS system, integrating
López de Prado's meta-labeling methodology with zenML pipelines for
institutional-grade signal validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

from .portfolio_analyzer import PortfolioAnalyzer
from ..pipelines.validation_pipeline import (
    asset_diagnostic_pipeline,
    batch_asset_diagnostics,
    asset_comparison_pipeline
)
from ..meta_labeling.primary_model import PrimaryDirectionModel
from ..meta_labeling.meta_model import MetaModel
from ..meta_labeling.triple_barrier import TripleBarrierLabeler


class SOROS:
    """
    Modern SOROS interface with meta-labeling and institutional-grade validation.
    
    This class provides a unified interface for:
    - Asset diagnostics with meta-labeling
    - 5-step validation framework (López de Prado methodology)
    - Asset comparison and benchmarking
    - ZenML-powered experiment tracking
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        asset_ids: Optional[List[str]] = None,
        use_meta_labeling: bool = True,
        primary_model_type: str = 'random_forest',
        meta_model_type: str = 'random_forest',
        experiment_name: Optional[str] = None,
        **legacy_kwargs
    ):
        """
        Initialize the enhanced SOROS system.
        
        Args:
            data_dir: Base directory for data storage
            asset_ids: List of asset IDs to work with
            use_meta_labeling: Whether to use meta-labeling framework
            primary_model_type: Type of primary model ('random_forest', 'gradient_boost', 'logistic')
            meta_model_type: Type of meta-model
            experiment_name: MLflow experiment name
            **legacy_kwargs: Additional arguments for backward compatibility
        """
        self.data_dir = data_dir
        self.asset_ids = asset_ids or []
        self.use_meta_labeling = use_meta_labeling
        self.primary_model_type = primary_model_type
        self.meta_model_type = meta_model_type
        self.experiment_name = experiment_name or f"soros_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize legacy PortfolioAnalyzer for backward compatibility
        self._legacy_analyzer = PortfolioAnalyzer(
            data_dir=data_dir,
            asset_ids=asset_ids,
            use_meta_labeling=use_meta_labeling,
            **legacy_kwargs
        )
        
        # Store validation results
        self.validation_results = {}
        self.diagnostic_reports = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"SOROS system initialized with meta-labeling: {use_meta_labeling}")
    
    def run_asset_diagnostics(
        self,
        asset_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        signal_names: Optional[List[str]] = None,
        validation_steps: int = 5,
        save_results: bool = True,
        **pipeline_kwargs
    ) -> Dict[str, Any]:
        """
        Run comprehensive asset diagnostics with meta-labeling validation.
        
        Args:
            asset_id: Asset to analyze
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            signal_names: List of signals to use (None for all)
            validation_steps: Number of validation steps (4 or 5)
            save_results: Whether to save results to instance
            **pipeline_kwargs: Additional pipeline arguments
            
        Returns:
            Comprehensive diagnostic report
        """
        self.logger.info(f"Running {validation_steps}-step diagnostics for {asset_id}")
        
        if validation_steps == 5 and self.use_meta_labeling:
            # Full 5-step validation with meta-labeling
            report = asset_diagnostic_pipeline(
                asset_id=asset_id,
                data_dir=self.data_dir,
                start_date=start_date,
                end_date=end_date,
                primary_model_type=self.primary_model_type,
                meta_model_type=self.meta_model_type,
                signal_names=signal_names,
                **pipeline_kwargs
            )
        else:
            # Fallback to legacy 4-step validation
            self.logger.warning("Falling back to legacy 4-step validation")
            report = self._legacy_validation(
                asset_id, start_date, end_date, signal_names
            )
        
        if save_results:
            self.diagnostic_reports[asset_id] = report
        
        return report
    
    def run_batch_diagnostics(
        self,
        asset_list: Optional[List[str]] = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run diagnostics for multiple assets in batch.
        
        Args:
            asset_list: List of assets to analyze (uses self.asset_ids if None)
            parallel: Whether to run in parallel
            **kwargs: Arguments passed to individual diagnostics
            
        Returns:
            Dictionary mapping asset IDs to diagnostic reports
        """
        assets_to_analyze = asset_list or self.asset_ids
        
        if not assets_to_analyze:
            raise ValueError("No assets specified for batch diagnostics")
        
        self.logger.info(f"Running batch diagnostics for {len(assets_to_analyze)} assets")
        
        if self.use_meta_labeling:
            results = batch_asset_diagnostics(
                asset_list=assets_to_analyze,
                data_dir=self.data_dir,
                **kwargs
            )
        else:
            # Legacy batch processing
            results = {}
            for asset_id in assets_to_analyze:
                try:
                    results[asset_id] = self.run_asset_diagnostics(
                        asset_id, save_results=False, **kwargs
                    )
                except Exception as e:
                    self.logger.error(f"Failed to analyze {asset_id}: {e}")
                    results[asset_id] = {'error': str(e), 'validation_passed': False}
        
        # Save batch results
        self.diagnostic_reports.update(results)
        
        return results
    
    def compare_assets(
        self,
        primary_asset: str,
        benchmark_asset: str = "bitcoin",
        comparison_metrics: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare asset performance against benchmark.
        
        Args:
            primary_asset: Primary asset to analyze
            benchmark_asset: Benchmark for comparison ("bitcoin", "ethereum", "usd")
            comparison_metrics: Metrics to compare
            **kwargs: Additional comparison arguments
            
        Returns:
            Comparative analysis report
        """
        self.logger.info(f"Comparing {primary_asset} vs {benchmark_asset}")
        
        if self.use_meta_labeling:
            comparison_report = asset_comparison_pipeline(
                primary_asset=primary_asset,
                benchmark_asset=benchmark_asset,
                data_dir=self.data_dir,
                **kwargs
            )
        else:
            # Legacy comparison using PortfolioAnalyzer
            comparison_report = self._legacy_comparison(
                primary_asset, benchmark_asset, comparison_metrics
            )
        
        return comparison_report
    
    def validate_signals(
        self,
        asset_id: str,
        signal_config: Dict[str, Any],
        validation_method: str = "meta_labeling"
    ) -> Dict[str, Any]:
        """
        Validate specific signal configuration using meta-labeling.
        
        Args:
            asset_id: Asset to test signals on
            signal_config: Signal configuration to validate
            validation_method: Validation method ("meta_labeling", "traditional")
            
        Returns:
            Signal validation results
        """
        self.logger.info(f"Validating signals for {asset_id} using {validation_method}")
        
        if validation_method == "meta_labeling" and self.use_meta_labeling:
            # Use meta-labeling validation
            return self.run_asset_diagnostics(
                asset_id=asset_id,
                signal_names=signal_config.get('signal_names'),
                labeling_params=signal_config.get('labeling_params'),
                param_grid=signal_config.get('param_grid')
            )
        else:
            # Use legacy validation
            return self._legacy_analyzer.validate_signal_config(asset_id, signal_config)
    
    def get_diagnostic_summary(
        self,
        asset_list: Optional[List[str]] = None,
        include_failed: bool = True
    ) -> pd.DataFrame:
        """
        Get summary of diagnostic results across assets.
        
        Args:
            asset_list: Assets to include in summary
            include_failed: Whether to include failed validations
            
        Returns:
            DataFrame with diagnostic summary
        """
        assets = asset_list or list(self.diagnostic_reports.keys())
        
        summary_data = []
        for asset_id in assets:
            if asset_id in self.diagnostic_reports:
                report = self.diagnostic_reports[asset_id]
                
                if not include_failed and not report.get('validation_passed', False):
                    continue
                
                summary_data.append({
                    'asset_id': asset_id,
                    'validation_passed': report.get('validation_passed', False),
                    'robustness_level': report.get('robustness_level', 'unknown'),
                    'oos_sharpe': report.get('oos_sharpe', 0),
                    'max_drawdown': report.get('max_drawdown', 0),
                    'signal_rate': report.get('signal_rate', 0),
                    'oos_p_value': report.get('oos_p_value', 1.0)
                })
        
        return pd.DataFrame(summary_data)
    
    def get_top_assets(
        self,
        metric: str = 'oos_sharpe',
        min_robustness: str = 'medium',
        top_k: int = 10
    ) -> List[str]:
        """
        Get top-performing assets based on validation results.
        
        Args:
            metric: Metric to rank by ('oos_sharpe', 'signal_rate', etc.)
            min_robustness: Minimum robustness level ('low', 'medium', 'high')
            top_k: Number of top assets to return
            
        Returns:
            List of top asset IDs
        """
        summary = self.get_diagnostic_summary()
        
        if summary.empty:
            return []
        
        # Filter by validation and robustness
        filtered = summary[
            (summary['validation_passed'] == True) &
            (summary['robustness_level'].isin(['medium', 'high'] if min_robustness == 'medium' 
                                             else ['high'] if min_robustness == 'high'
                                             else ['low', 'medium', 'high']))
        ]
        
        if filtered.empty:
            return []
        
        # Sort by metric and return top K
        top_assets = filtered.nlargest(top_k, metric)['asset_id'].tolist()
        
        return top_assets
    
    def export_results(
        self,
        output_dir: str = 'output',
        format: str = 'json',
        include_charts: bool = True
    ) -> str:
        """
        Export diagnostic results to files.
        
        Args:
            output_dir: Directory to save results
            format: Export format ('json', 'csv', 'excel')
            include_charts: Whether to include charts
            
        Returns:
            Path to exported results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            import json
            output_path = os.path.join(output_dir, f'soros_results_{timestamp}.json')
            with open(output_path, 'w') as f:
                json.dump(self.diagnostic_reports, f, indent=2, default=str)
        
        elif format == 'csv':
            output_path = os.path.join(output_dir, f'soros_summary_{timestamp}.csv')
            summary = self.get_diagnostic_summary()
            summary.to_csv(output_path, index=False)
        
        elif format == 'excel':
            output_path = os.path.join(output_dir, f'soros_results_{timestamp}.xlsx')
            summary = self.get_diagnostic_summary()
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add individual asset sheets
                for asset_id, report in self.diagnostic_reports.items():
                    if isinstance(report, dict):
                        asset_df = pd.DataFrame([report])
                        asset_df.to_excel(writer, sheet_name=asset_id[:30], index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Results exported to: {output_path}")
        return output_path
    
    # Legacy compatibility methods
    
    def _legacy_validation(
        self,
        asset_id: str,
        start_date: Optional[str],
        end_date: Optional[str],
        signal_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Legacy 4-step validation using PortfolioAnalyzer."""
        try:
            results = self._legacy_analyzer.run_validation_pipeline(
                asset_id, start_date, end_date, signal_names
            )
            return {
                'asset_id': asset_id,
                'validation_passed': results.get('validation_passed', False),
                'oos_sharpe': results.get('oos_sharpe', 0),
                'legacy_mode': True
            }
        except Exception as e:
            self.logger.error(f"Legacy validation failed for {asset_id}: {e}")
            return {
                'asset_id': asset_id,
                'validation_passed': False,
                'error': str(e),
                'legacy_mode': True
            }
    
    def _legacy_comparison(
        self,
        primary_asset: str,
        benchmark_asset: str,
        comparison_metrics: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Legacy comparison using PortfolioAnalyzer."""
        try:
            primary_results = self._legacy_analyzer.analyze_asset(primary_asset)
            benchmark_results = self._legacy_analyzer.analyze_asset(benchmark_asset)
            
            return {
                'primary_asset': primary_asset,
                'benchmark_asset': benchmark_asset,
                'primary_results': primary_results,
                'benchmark_results': benchmark_results,
                'legacy_mode': True
            }
        except Exception as e:
            self.logger.error(f"Legacy comparison failed: {e}")
            return {'error': str(e), 'legacy_mode': True}
    
    # Backward compatibility properties
    
    @property
    def data_dir(self):
        """Data directory path."""
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, value):
        self._data_dir = value
        if hasattr(self, '_legacy_analyzer'):
            self._legacy_analyzer.data_dir = value
    
    def load_data(self, asset_ids: List[str], start_date: str = None, end_date: str = None):
        """Load data for assets (backward compatibility)."""
        self.asset_ids = asset_ids
        return self._legacy_analyzer.load_data(asset_ids, start_date, end_date)
    
    def get_signals(self, asset_id: str, signal_names: List[str] = None):
        """Get signals for asset (backward compatibility)."""
        return self._legacy_analyzer.get_signals(asset_id, signal_names)


# Factory function for easy migration
def create_soros(
    data_dir: str = 'data',
    asset_ids: Optional[List[str]] = None,
    enable_meta_labeling: bool = True,
    **kwargs
) -> SOROS:
    """
    Factory function to create SOROS instance with sensible defaults.
    
    Args:
        data_dir: Data directory path
        asset_ids: List of asset IDs
        enable_meta_labeling: Enable López de Prado's meta-labeling
        **kwargs: Additional arguments
        
    Returns:
        Configured SOROS instance
    """
    return SOROS(
        data_dir=data_dir,
        asset_ids=asset_ids,
        use_meta_labeling=enable_meta_labeling,
        **kwargs
    )