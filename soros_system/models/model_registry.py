"""
Model registry and versioning system for SOROS meta-labeling framework.

This module provides comprehensive model lifecycle management including:
- Model versioning with semantic versioning
- Performance tracking and comparison
- Model selection and deployment
- Cross-asset model management
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import mlflow
import mlflow.sklearn

from .base_primary_model import BasePrimaryModel
from .primary_ml import MLBasedPrimaryModel
from .primary_rules import RuleBasedPrimaryModel
from .primary_hybrid import HybridPrimaryModel


@dataclass
class ModelMetadata:
    """Model metadata for registry."""
    model_id: str
    asset_id: str
    model_type: str  # 'ml_based', 'rule_based', 'hybrid'
    model_subtype: str  # 'random_forest', 'ensemble', 'adaptive', etc.
    version: str
    created_at: str
    performance_metrics: Dict[str, float]
    file_path: str
    config_path: Optional[str] = None
    training_data_hash: Optional[str] = None
    is_production: bool = False
    tags: Optional[Dict[str, str]] = None


class ModelRegistry:
    """
    Centralized model registry for managing model versions and deployments.
    
    Provides functionality for:
    - Model registration and versioning
    - Performance tracking and comparison
    - Model selection based on performance
    - Production deployment management
    """
    
    def __init__(
        self,
        registry_dir: str = 'models',
        experiment_name: str = 'soros_model_registry'
    ):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store model files
            experiment_name: MLflow experiment name
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_dir / 'registry_metadata.json'
        self.experiment_name = experiment_name
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        # Load existing metadata
        self._metadata_cache = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """Load model metadata from registry file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Convert dict to ModelMetadata objects
            metadata_cache = {}
            for model_id, metadata_dict in data.items():
                metadata_cache[model_id] = ModelMetadata(**metadata_dict)
            
            return metadata_cache
            
        except Exception as e:
            self.logger.warning(f"Failed to load model metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save model metadata to registry file."""
        try:
            # Convert ModelMetadata objects to dict
            data = {}
            for model_id, metadata in self._metadata_cache.items():
                data[model_id] = asdict(metadata)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model metadata: {e}")
    
    def register_model(
        self,
        model: BasePrimaryModel,
        asset_id: str,
        performance_metrics: Dict[str, float],
        model_subtype: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        training_data_hash: Optional[str] = None
    ) -> str:
        """
        Register a new model version in the registry.
        
        Args:
            model: Trained model instance
            asset_id: Asset identifier
            performance_metrics: Model performance metrics
            model_subtype: Subtype of model (e.g., 'random_forest', 'ensemble')
            tags: Additional tags for model
            training_data_hash: Hash of training data for reproducibility
            
        Returns:
            Model ID for the registered model
        """
        # Generate model ID
        model_id = self._generate_model_id(asset_id, model.model_type, model.model_version)
        
        # Create model directory
        model_dir = self.registry_dir / asset_id / model.model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model files
        model_file = model_dir / f"{model_id}.pkl"
        config_file = model_dir / f"{model_id}_config.json"
        
        # Save model based on type
        if hasattr(model, 'save_model'):
            model.save_model(str(model_file))
        else:
            # For rule-based models, save configuration
            model.save_config(str(config_file))
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            asset_id=asset_id,
            model_type=model.model_type,
            model_subtype=model_subtype or model.model_type,
            version=model.model_version,
            created_at=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            file_path=str(model_file),
            config_path=str(config_file) if config_file.exists() else None,
            training_data_hash=training_data_hash,
            tags=tags or {}
        )
        
        # Register with MLflow
        with mlflow.start_run(run_name=f"register_{model_id}"):
            mlflow.log_params({
                'model_id': model_id,
                'asset_id': asset_id,
                'model_type': model.model_type,
                'model_version': model.model_version
            })
            mlflow.log_metrics(performance_metrics)
            
            if model.model_type in ['ml_based', 'hybrid']:
                # Log ML model
                try:
                    if hasattr(model, 'model') and model.model is not None:
                        mlflow.sklearn.log_model(model.model, f"model_{model_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to log ML model to MLflow: {e}")
        
        # Add to cache and save
        self._metadata_cache[model_id] = metadata
        self._save_metadata()
        
        self.logger.info(f"Model registered: {model_id}")
        return model_id
    
    def get_model_versions(
        self,
        asset_id: str,
        model_type: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        Get all model versions for an asset.
        
        Args:
            asset_id: Asset identifier
            model_type: Filter by model type (optional)
            
        Returns:
            List of model metadata sorted by creation date
        """
        versions = []
        
        for metadata in self._metadata_cache.values():
            if metadata.asset_id == asset_id:
                if model_type is None or metadata.model_type == model_type:
                    versions.append(metadata)
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        return versions
    
    def get_best_model(
        self,
        asset_id: str,
        metric: str = 'oos_sharpe',
        model_type: Optional[str] = None,
        min_version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get the best performing model for an asset.
        
        Args:
            asset_id: Asset identifier
            metric: Performance metric to optimize
            model_type: Filter by model type
            min_version: Minimum model version to consider
            
        Returns:
            Best model metadata or None if no models found
        """
        versions = self.get_model_versions(asset_id, model_type)
        
        if not versions:
            return None
        
        # Filter by minimum version if specified
        if min_version:
            versions = [v for v in versions if self._compare_versions(v.version, min_version) >= 0]
        
        # Find best model by metric
        best_model = None
        best_score = -np.inf
        
        for version in versions:
            score = version.performance_metrics.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model = version
        
        return best_model
    
    def load_model(self, model_id: str) -> Optional[BasePrimaryModel]:
        """
        Load a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model instance or None if not found
        """
        if model_id not in self._metadata_cache:
            self.logger.error(f"Model not found: {model_id}")
            return None
        
        metadata = self._metadata_cache[model_id]
        
        try:
            # Load based on model type
            if metadata.model_type == 'ml_based':
                return MLBasedPrimaryModel.load_model(metadata.file_path)
            elif metadata.model_type == 'rule_based':
                return RuleBasedPrimaryModel.load_config(metadata.config_path)
            elif metadata.model_type == 'hybrid':
                return HybridPrimaryModel.load_model(metadata.file_path)
            else:
                self.logger.error(f"Unknown model type: {metadata.model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def set_production_model(
        self,
        asset_id: str,
        model_id: str,
        model_type: str = 'primary'
    ) -> bool:
        """
        Set a model as the production model for an asset.
        
        Args:
            asset_id: Asset identifier
            model_id: Model to promote to production
            model_type: Type of model ('primary' or 'meta')
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self._metadata_cache:
            self.logger.error(f"Model not found: {model_id}")
            return False
        
        # Remove production flag from other models of same asset/type
        for metadata in self._metadata_cache.values():
            if (metadata.asset_id == asset_id and 
                metadata.tags.get('deployment_type') == model_type):
                metadata.is_production = False
                if metadata.tags is None:
                    metadata.tags = {}
                metadata.tags.pop('production_since', None)
        
        # Set new production model
        metadata = self._metadata_cache[model_id]
        metadata.is_production = True
        if metadata.tags is None:
            metadata.tags = {}
        metadata.tags['deployment_type'] = model_type
        metadata.tags['production_since'] = datetime.now().isoformat()
        
        # Save changes
        self._save_metadata()
        
        self.logger.info(f"Set production model for {asset_id}: {model_id}")
        return True
    
    def get_production_model(
        self,
        asset_id: str,
        model_type: str = 'primary'
    ) -> Optional[Tuple[BasePrimaryModel, ModelMetadata]]:
        """
        Get the current production model for an asset.
        
        Args:
            asset_id: Asset identifier
            model_type: Type of model ('primary' or 'meta')
            
        Returns:
            Tuple of (model_instance, metadata) or None if not found
        """
        # Find production model
        for metadata in self._metadata_cache.values():
            if (metadata.asset_id == asset_id and 
                metadata.is_production and
                metadata.tags.get('deployment_type') == model_type):
                
                model = self.load_model(metadata.model_id)
                if model:
                    return model, metadata
        
        return None
    
    def compare_models(
        self,
        asset_id: str,
        metrics: List[str] = None,
        model_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare model performance across versions.
        
        Args:
            asset_id: Asset identifier
            metrics: Metrics to include in comparison
            model_types: Model types to include
            
        Returns:
            DataFrame with model comparison
        """
        if metrics is None:
            metrics = ['oos_sharpe', 'max_drawdown', 'signal_rate', 'oos_p_value']
        
        versions = self.get_model_versions(asset_id)
        
        if model_types:
            versions = [v for v in versions if v.model_type in model_types]
        
        comparison_data = []
        for version in versions:
            row = {
                'model_id': version.model_id,
                'model_type': version.model_type,
                'version': version.version,
                'created_at': version.created_at,
                'is_production': version.is_production
            }
            
            # Add performance metrics
            for metric in metrics:
                row[metric] = version.performance_metrics.get(metric, np.nan)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_models(
        self,
        asset_id: str,
        keep_versions: int = 5,
        keep_production: bool = True
    ) -> int:
        """
        Clean up old model versions to save space.
        
        Args:
            asset_id: Asset identifier
            keep_versions: Number of recent versions to keep
            keep_production: Whether to always keep production models
            
        Returns:
            Number of models deleted
        """
        versions = self.get_model_versions(asset_id)
        
        # Determine which models to delete
        to_delete = []
        kept_count = 0
        
        for version in versions:
            should_keep = (
                kept_count < keep_versions or
                (keep_production and version.is_production)
            )
            
            if should_keep:
                kept_count += 1
            else:
                to_delete.append(version)
        
        # Delete models
        deleted_count = 0
        for version in to_delete:
            try:
                # Remove files
                if os.path.exists(version.file_path):
                    os.remove(version.file_path)
                if version.config_path and os.path.exists(version.config_path):
                    os.remove(version.config_path)
                
                # Remove from cache
                del self._metadata_cache[version.model_id]
                deleted_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to delete model {version.model_id}: {e}")
        
        # Save updated metadata
        if deleted_count > 0:
            self._save_metadata()
            self.logger.info(f"Cleaned up {deleted_count} old models for {asset_id}")
        
        return deleted_count
    
    def export_model_registry(self, output_path: str) -> None:
        """Export complete model registry to file."""
        registry_data = {
            'metadata': {model_id: asdict(metadata) for model_id, metadata in self._metadata_cache.items()},
            'export_timestamp': datetime.now().isoformat(),
            'registry_dir': str(self.registry_dir)
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Registry exported to {output_path}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            'total_models': len(self._metadata_cache),
            'models_by_type': {},
            'models_by_asset': {},
            'production_models': 0,
            'registry_size_mb': 0
        }
        
        # Calculate statistics
        for metadata in self._metadata_cache.values():
            # By type
            model_type = metadata.model_type
            stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1
            
            # By asset
            asset_id = metadata.asset_id
            stats['models_by_asset'][asset_id] = stats['models_by_asset'].get(asset_id, 0) + 1
            
            # Production models
            if metadata.is_production:
                stats['production_models'] += 1
            
            # File sizes
            if os.path.exists(metadata.file_path):
                stats['registry_size_mb'] += os.path.getsize(metadata.file_path) / (1024 * 1024)
        
        return stats
    
    def _generate_model_id(self, asset_id: str, model_type: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{asset_id}_{model_type}_v{version}_{timestamp}"
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two semantic versions. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad to same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] > v2_parts[i]:
                    return 1
                elif v1_parts[i] < v2_parts[i]:
                    return -1
            
            return 0
        except Exception:
            # Fallback to string comparison
            if version1 > version2:
                return 1
            elif version1 < version2:
                return -1
            else:
                return 0


# Global registry instance
_global_registry = None


def get_model_registry(registry_dir: str = 'models') -> ModelRegistry:
    """Get or create global model registry instance."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ModelRegistry(registry_dir)
    
    return _global_registry


def register_model(
    model: BasePrimaryModel,
    asset_id: str,
    performance_metrics: Dict[str, float],
    **kwargs
) -> str:
    """Convenience function to register a model."""
    registry = get_model_registry()
    return registry.register_model(model, asset_id, performance_metrics, **kwargs)


def get_best_model_for_asset(
    asset_id: str,
    metric: str = 'oos_sharpe',
    model_type: Optional[str] = None
) -> Optional[BasePrimaryModel]:
    """Convenience function to get best model for an asset."""
    registry = get_model_registry()
    best_metadata = registry.get_best_model(asset_id, metric, model_type)
    
    if best_metadata:
        return registry.load_model(best_metadata.model_id)
    
    return None