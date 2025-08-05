"""
Enhanced SOROS interface with comprehensive model management.

This module provides the modern interface for the SOROS system with support for:
- Multiple primary model types (ML-based, rule-based, hybrid)
- Advanced model registry and versioning
- Performance monitoring and A/B testing
- Team collaboration features
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

from .portfolio_analyzer import PortfolioAnalyzer
from ..pipelines.enhanced_validation_pipeline import (
    enhanced_asset_diagnostic_pipeline,
    model_comparison_pipeline,
    batch_enhanced_diagnostics
)
from ..models.model_registry import get_model_registry, ModelRegistry
from ..models.primary_ml import MLBasedPrimaryModel
from ..models.primary_rules import RuleBasedPrimaryModel
from ..models.primary_hybrid import HybridPrimaryModel


class EnhancedSOROS:
    """
    Enhanced SOROS interface with comprehensive model management.
    
    This class provides advanced functionality including:
    - Multiple primary model types (ML-based, rule-based, hybrid)
    - Model registry with versioning and performance tracking
    - A/B testing and model comparison
    - Performance monitoring and automated model selection
    - Team collaboration with Git-based sharing
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        model_registry_dir: str = 'models',
        asset_ids: Optional[List[str]] = None,
        default_primary_model_mode: str = 'hybrid',
        default_meta_model_type: str = 'random_forest',
        experiment_name: Optional[str] = None,
        enable_model_registry: bool = True,
        **legacy_kwargs
    ):
        """
        Initialize the enhanced SOROS system.
        
        Args:
            data_dir: Base directory for data storage
            model_registry_dir: Directory for model registry
            asset_ids: List of asset IDs to work with
            default_primary_model_mode: Default primary model type ('ml_based', 'rule_based', 'hybrid')
            default_meta_model_type: Default meta-model type
            experiment_name: MLflow experiment name
            enable_model_registry: Whether to use model registry
            **legacy_kwargs: Backward compatibility arguments
        """
        self.data_dir = data_dir
        self.model_registry_dir = model_registry_dir
        self.asset_ids = asset_ids or []
        self.default_primary_model_mode = default_primary_model_mode
        self.default_meta_model_type = default_meta_model_type
        self.experiment_name = experiment_name or f"enhanced_soros_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_model_registry = enable_model_registry
        
        # Initialize model registry
        if enable_model_registry:
            self.model_registry = get_model_registry(model_registry_dir)
        else:
            self.model_registry = None
        
        # Initialize legacy PortfolioAnalyzer for backward compatibility
        # Filter out parameters that might conflict
        filtered_kwargs = {k: v for k, v in legacy_kwargs.items() 
                          if k not in ['default_primary_model_mode', 'default_meta_model_type', 
                                      'enable_model_registry', 'model_registry_dir']}
        self._legacy_analyzer = PortfolioAnalyzer(
            data_dir=data_dir,
            asset_ids=asset_ids,
            **filtered_kwargs
        )
        
        # Store diagnostic results and model information
        self.diagnostic_results = {}
        self.model_comparison_results = {}
        self.active_models = {}  # Asset -> Model metadata mapping
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"Enhanced SOROS initialized with primary model mode: {default_primary_model_mode}, "
            f"registry enabled: {enable_model_registry}"
        )
    
    def run_asset_diagnostics(
        self,
        asset_id: str,
        primary_model_mode: Optional[str] = None,
        primary_model_config: Optional[Dict[str, Any]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validation_steps: int = 5,
        use_existing_model: bool = True,
        register_model: bool = None,
        save_results: bool = True,
        **pipeline_kwargs
    ) -> Dict[str, Any]:
        """
        Run enhanced asset diagnostics with configurable model types.
        
        Args:
            asset_id: Asset to analyze
            primary_model_mode: Primary model type ('ml_based', 'rule_based', 'hybrid')
            primary_model_config: Configuration for primary model
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            validation_steps: Number of validation steps (always 5 for enhanced)
            use_existing_model: Whether to try loading existing models
            register_model: Whether to register models (None = use default)
            save_results: Whether to save results to instance
            **pipeline_kwargs: Additional pipeline arguments
            
        Returns:
            Enhanced diagnostic report
        """
        # Use defaults if not specified
        model_mode = primary_model_mode or self.default_primary_model_mode
        register = register_model if register_model is not None else self.enable_model_registry
        
        self.logger.info(f"Running enhanced diagnostics for {asset_id} with {model_mode} model")
        
        try:
            # For now, use SimpleRunner as a bridge while we fix ZenML pipeline execution
            # This maintains the same interface and features
            from .simple_runner import SimpleRunner
            runner = SimpleRunner(data_dir=self.data_dir)
            
            if model_mode == 'rule_based':
                # Run rule-based model
                result = runner.run_rule_based_model(
                    asset_id=asset_id,
                    signal_names=primary_model_config.get('signal_names') if primary_model_config else None,
                    combination_strategy=primary_model_config.get('combination_strategy', 'ensemble') if primary_model_config else 'ensemble',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if result['success']:
                    # Register model if enabled
                    primary_model_id = None
                    if register and self.model_registry:
                        primary_model_id = self.model_registry.register_model(
                            model=result['model'],
                            asset_id=asset_id,
                            performance_metrics={
                                'sharpe_ratio': result.get('sharpe_ratio', 0),
                                'total_return': result.get('total_return', 0),
                                'max_drawdown': result.get('max_drawdown', 0),
                                'signal_rate': result.get('signal_rate', 0)
                            }
                        )
                    
                    report = {
                        'asset_id': asset_id,
                        'primary_model_type': 'rule_based',
                        'primary_model_version': result['model_version'],
                        'primary_model_id': primary_model_id,
                        'meta_model_type': self.default_meta_model_type,
                        'validation_passed': result.get('sharpe_ratio', 0) > 0,
                        'oos_sharpe': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'signal_rate': result.get('signal_rate', 0),
                        'primary_training_metrics': {'training_required': False},
                        'meta_training_metrics': {}
                    }
                else:
                    raise RuntimeError(result['error'])
            else:
                # For ML-based and hybrid, we'll use the full pipeline when ZenML is properly configured
                # For now, return a placeholder
                self.logger.warning(f"ML-based and hybrid models require ZenML pipeline - using placeholder")
                report = {
                    'asset_id': asset_id,
                    'primary_model_type': model_mode,
                    'validation_passed': False,
                    'oos_sharpe': 0,
                    'error': 'ZenML pipeline integration in progress'
                }
            
            # Store active model information
            if register and report.get('primary_model_id'):
                self.active_models[asset_id] = {
                    'model_id': report['primary_model_id'],
                    'model_mode': model_mode,
                    'performance': report.get('oos_sharpe', 0),
                    'timestamp': datetime.now().isoformat()
                }
            
            if save_results:
                self.diagnostic_results[asset_id] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"Enhanced diagnostics failed for {asset_id}: {e}")
            # Fallback to legacy validation if enhanced fails
            return self._fallback_legacy_validation(
                asset_id, start_date, end_date, model_mode
            )
    
    def compare_model_types(
        self,
        asset_id: str,
        model_modes: Optional[List[str]] = None,
        comparison_metric: str = 'oos_sharpe',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare different model types for the same asset.
        
        Args:
            asset_id: Asset to analyze
            model_modes: List of model modes to compare
            comparison_metric: Metric to use for comparison
            **kwargs: Additional arguments for diagnostics
            
        Returns:
            Model comparison results
        """
        if model_modes is None:
            model_modes = ['ml_based', 'rule_based', 'hybrid']
        
        self.logger.info(f"Comparing model types for {asset_id}: {model_modes}")
        
        try:
            comparison_results = model_comparison_pipeline(
                asset_id=asset_id,
                model_modes=model_modes,
                data_dir=self.data_dir,
                comparison_metric=comparison_metric,
                **kwargs
            )
            
            # Store comparison results
            self.model_comparison_results[asset_id] = comparison_results
            
            # Update active model to best performing
            best_mode = comparison_results.get('best_model_mode')
            if best_mode and self.enable_model_registry:
                best_report = comparison_results['model_results'].get(best_mode, {})
                if best_report.get('primary_model_id'):
                    self.active_models[asset_id] = {
                        'model_id': best_report['primary_model_id'],
                        'model_mode': best_mode,
                        'performance': best_report.get('oos_sharpe', 0),
                        'timestamp': datetime.now().isoformat(),
                        'comparison_winner': True
                    }
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Model comparison failed for {asset_id}: {e}")
            return {'error': str(e), 'asset_id': asset_id}
    
    def run_batch_diagnostics(
        self,
        asset_list: Optional[List[str]] = None,
        primary_model_mode: Optional[str] = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run enhanced diagnostics for multiple assets.
        
        Args:
            asset_list: List of assets to analyze (uses self.asset_ids if None)
            primary_model_mode: Primary model mode for all assets
            parallel: Whether to run in parallel
            **kwargs: Additional arguments
            
        Returns:
            Batch diagnostic results
        """
        assets_to_analyze = asset_list or self.asset_ids
        model_mode = primary_model_mode or self.default_primary_model_mode
        
        if not assets_to_analyze:
            raise ValueError("No assets specified for batch diagnostics")
        
        self.logger.info(
            f"Running batch enhanced diagnostics for {len(assets_to_analyze)} assets "
            f"with {model_mode} models"
        )
        
        try:
            results = batch_enhanced_diagnostics(
                asset_list=assets_to_analyze,
                primary_model_mode=model_mode,
                data_dir=self.data_dir,
                parallel=parallel,
                **kwargs
            )
            
            # Update stored results and active models
            for asset_id, report in results.items():
                if not report.get('error'):
                    self.diagnostic_results[asset_id] = report
                    
                    if report.get('primary_model_id'):
                        self.active_models[asset_id] = {
                            'model_id': report['primary_model_id'],
                            'model_mode': model_mode,
                            'performance': report.get('oos_sharpe', 0),
                            'timestamp': datetime.now().isoformat()
                        }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch diagnostics failed: {e}")
            return {'error': str(e)}
    
    def get_model_performance_summary(
        self,
        asset_list: Optional[List[str]] = None,
        include_model_details: bool = True
    ) -> pd.DataFrame:
        """
        Get performance summary across assets and model types.
        
        Args:
            asset_list: Assets to include in summary
            include_model_details: Whether to include model-specific details
            
        Returns:
            DataFrame with model performance summary
        """
        if not self.enable_model_registry:
            return self.get_diagnostic_summary(asset_list)
        
        assets = asset_list or list(self.active_models.keys())
        summary_data = []
        
        for asset_id in assets:
            # Get diagnostic results
            if asset_id in self.diagnostic_results:
                report = self.diagnostic_results[asset_id]
                
                row = {
                    'asset_id': asset_id,
                    'validation_passed': report.get('validation_passed', False),
                    'robustness_level': report.get('robustness_level', 'unknown'),
                    'oos_sharpe': report.get('oos_sharpe', 0),
                    'max_drawdown': report.get('max_drawdown', 0),
                    'signal_rate': report.get('signal_rate', 0),
                    'oos_p_value': report.get('oos_p_value', 1.0)
                }
                
                if include_model_details:
                    row.update({
                        'primary_model_type': report.get('primary_model_type', 'unknown'),
                        'primary_model_version': report.get('primary_model_version', '1.0.0'),
                        'meta_model_type': report.get('meta_model_type', 'unknown'),
                        'model_registered': report.get('primary_model_id') is not None
                    })
                
                # Add active model info
                if asset_id in self.active_models:
                    active_info = self.active_models[asset_id]
                    row.update({
                        'active_model_id': active_info.get('model_id'),
                        'active_model_mode': active_info.get('model_mode'),
                        'last_updated': active_info.get('timestamp')
                    })
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_models_by_metric(
        self,
        metric: str = 'oos_sharpe',
        min_robustness: str = 'medium',
        top_k: int = 10,
        model_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get best performing models across all assets.
        
        Args:
            metric: Metric to rank by
            min_robustness: Minimum robustness level
            top_k: Number of top models to return
            model_type_filter: Filter by primary model type
            
        Returns:
            List of best model information
        """
        summary = self.get_model_performance_summary()
        
        if summary.empty:
            return []
        
        # Filter by validation and robustness
        robustness_levels = {
            'low': ['low', 'medium', 'high'],
            'medium': ['medium', 'high'],
            'high': ['high']
        }
        
        filtered = summary[
            (summary['validation_passed'] == True) &
            (summary['robustness_level'].isin(robustness_levels.get(min_robustness, ['high'])))
        ]
        
        # Filter by model type if specified
        if model_type_filter and 'primary_model_type' in filtered.columns:
            filtered = filtered[filtered['primary_model_type'] == model_type_filter]
        
        if filtered.empty:
            return []
        
        # Sort and get top K
        top_models = filtered.nlargest(top_k, metric)
        
        return top_models.to_dict('records')
    
    def deploy_model_to_production(
        self,
        asset_id: str,
        model_id: Optional[str] = None,
        model_type: str = 'primary'
    ) -> bool:
        """
        Deploy a model to production for an asset.
        
        Args:
            asset_id: Asset identifier
            model_id: Model to deploy (uses best if None)
            model_type: Type of model deployment
            
        Returns:
            True if successful
        """
        if not self.enable_model_registry:
            self.logger.warning("Model registry disabled - cannot deploy to production")
            return False
        
        if model_id is None:
            # Use best available model
            best_metadata = self.model_registry.get_best_model(asset_id)
            if best_metadata:
                model_id = best_metadata.model_id
            else:
                self.logger.error(f"No models available for {asset_id}")
                return False
        
        success = self.model_registry.set_production_model(asset_id, model_id, model_type)
        
        if success:
            self.logger.info(f"Deployed model {model_id} to production for {asset_id}")
            
            # Update active models
            if asset_id in self.active_models:
                self.active_models[asset_id]['is_production'] = True
                self.active_models[asset_id]['production_since'] = datetime.now().isoformat()
        
        return success
    
    def get_model_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive model registry statistics."""
        if not self.enable_model_registry:
            return {'error': 'Model registry disabled'}
        
        # Get registry stats
        stats = self.model_registry.get_registry_stats()
        
        # Add SOROS-specific stats
        stats.update({
            'active_models': len(self.active_models),
            'diagnostics_completed': len(self.diagnostic_results),
            'model_comparisons_run': len(self.model_comparison_results),
            'assets_with_production_models': sum(
                1 for model_info in self.active_models.values()
                if model_info.get('is_production', False)
            )
        })
        
        return stats
    
    def cleanup_old_models(
        self,
        asset_id: Optional[str] = None,
        keep_versions: int = 5,
        keep_production: bool = True
    ) -> Dict[str, int]:
        """
        Clean up old model versions.
        
        Args:
            asset_id: Specific asset (None for all)
            keep_versions: Versions to keep per asset
            keep_production: Whether to always keep production models
            
        Returns:
            Dictionary of cleanup statistics
        """
        if not self.enable_model_registry:
            return {'error': 'Model registry disabled'}
        
        cleanup_stats = {}
        
        if asset_id:
            # Clean up specific asset
            deleted = self.model_registry.cleanup_old_models(
                asset_id, keep_versions, keep_production
            )
            cleanup_stats[asset_id] = deleted
        else:
            # Clean up all assets
            assets_to_clean = set(self.asset_ids + list(self.active_models.keys()))
            
            for asset in assets_to_clean:
                try:
                    deleted = self.model_registry.cleanup_old_models(
                        asset, keep_versions, keep_production
                    )
                    cleanup_stats[asset] = deleted
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup models for {asset}: {e}")
                    cleanup_stats[asset] = 0
        
        total_deleted = sum(cleanup_stats.values())
        self.logger.info(f"Cleaned up {total_deleted} old models")
        
        return cleanup_stats
    
    def export_enhanced_results(
        self,
        output_dir: str = 'output',
        format: str = 'json',
        include_model_registry: bool = True
    ) -> str:
        """
        Export enhanced results including model information.
        
        Args:
            output_dir: Directory to save results
            format: Export format
            include_model_registry: Whether to include model registry export
            
        Returns:
            Path to exported results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare enhanced export data
        export_data = {
            'diagnostic_results': self.diagnostic_results,
            'model_comparison_results': self.model_comparison_results,
            'active_models': self.active_models,
            'model_registry_stats': self.get_model_registry_stats() if self.enable_model_registry else {},
            'configuration': {
                'default_primary_model_mode': self.default_primary_model_mode,
                'default_meta_model_type': self.default_meta_model_type,
                'model_registry_enabled': self.enable_model_registry
            },
            'export_timestamp': timestamp
        }
        
        if format == 'json':
            import json
            output_path = os.path.join(output_dir, f'enhanced_soros_results_{timestamp}.json')
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format == 'csv':
            output_path = os.path.join(output_dir, f'enhanced_soros_summary_{timestamp}.csv')
            summary = self.get_model_performance_summary()
            summary.to_csv(output_path, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Export model registry separately if requested
        if include_model_registry and self.enable_model_registry:
            registry_path = os.path.join(output_dir, f'model_registry_{timestamp}.json')
            self.model_registry.export_model_registry(registry_path)
        
        self.logger.info(f"Enhanced results exported to: {output_path}")
        return output_path
    
    def _fallback_legacy_validation(
        self,
        asset_id: str,
        start_date: Optional[str],
        end_date: Optional[str],
        model_mode: str
    ) -> Dict[str, Any]:
        """Fallback to legacy validation when enhanced pipeline fails."""
        try:
            self.logger.warning(f"Using legacy validation for {asset_id}")
            results = self._legacy_analyzer.run_validation_pipeline(
                asset_id, start_date, end_date, None
            )
            
            # Convert to enhanced format
            return {
                'asset_id': asset_id,
                'validation_passed': results.get('validation_passed', False),
                'oos_sharpe': results.get('oos_sharpe', 0),
                'primary_model_type': 'legacy',
                'legacy_mode': True,
                'fallback_reason': 'Enhanced pipeline failed'
            }
        except Exception as e:
            self.logger.error(f"Legacy validation also failed for {asset_id}: {e}")
            return {
                'asset_id': asset_id,
                'validation_passed': False,
                'error': str(e),
                'legacy_mode': True
            }
    
    # Legacy compatibility methods
    
    def load_data(self, asset_ids: List[str], start_date: str = None, end_date: str = None):
        """Load data for assets (backward compatibility)."""
        self.asset_ids = asset_ids
        return self._legacy_analyzer.load_data(asset_ids, start_date, end_date)
    
    def get_signals(self, asset_id: str, signal_names: List[str] = None):
        """Get signals for asset (backward compatibility)."""
        return self._legacy_analyzer.get_signals(asset_id, signal_names)


# Factory function for enhanced SOROS
def create_enhanced_soros(
    data_dir: str = 'data',
    model_registry_dir: str = 'models',
    asset_ids: Optional[List[str]] = None,
    default_primary_model_mode: str = 'hybrid',
    enable_model_registry: bool = True,
    **kwargs
) -> EnhancedSOROS:
    """
    Factory function to create enhanced SOROS instance.
    
    Args:
        data_dir: Data directory path
        model_registry_dir: Model registry directory
        asset_ids: List of asset IDs
        default_primary_model_mode: Default primary model mode
        enable_model_registry: Enable model registry and versioning
        **kwargs: Additional arguments
        
    Returns:
        Configured enhanced SOROS instance
    """
    return EnhancedSOROS(
        data_dir=data_dir,
        model_registry_dir=model_registry_dir,
        asset_ids=asset_ids,
        default_primary_model_mode=default_primary_model_mode,
        enable_model_registry=enable_model_registry,
        **kwargs
    )