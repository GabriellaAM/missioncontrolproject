"""
Signal evaluation storage for Soros System.

This module provides functionality for persisting signal evaluations
to disk and loading them back, preventing unnecessary recalculation
of signal effectiveness metrics.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Union
import pandas as pd


class SignalEvaluationStorage:
    """
    Storage for signal evaluation results.
    
    This class manages the persistence of signal evaluations, allowing
    signal selection decisions to be reused without recalculation.
    """
    
    def __init__(
        self,
        storage_dir: str = 'data/signal_evaluations',
        filename: str = 'signal_evaluations.json',
        version: str = '1.0'
    ):
        """Initialize the signal evaluation storage.
        
        Args:
            storage_dir: Directory for storing signal evaluations
            filename: Name of the evaluations file
            version: Version of the storage format
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = storage_dir
        self.filename = filename
        self.version = version
        self.filepath = os.path.join(storage_dir, filename)
        
        # Storage for evaluations
        self.evaluations = {
            "metadata": {
                "version": version,
                "last_updated": datetime.now().isoformat(),
                "num_assets": 0,
                "num_evaluations": 0
            },
            "assets": {}  # {asset_id: {signal_name: evaluation}}
        }
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            self.logger.info(f"Created signal evaluation storage directory: {storage_dir}")
    
    def load(self) -> bool:
        """Load signal evaluations from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.filepath):
            self.logger.warning(f"Signal evaluation file not found: {self.filepath}")
            return False
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            # Validate the loaded data
            if not self._validate_loaded_data(data):
                self.logger.error(f"Invalid signal evaluation data: {self.filepath}")
                return False
            
            # Store the loaded evaluations
            self.evaluations = data
            
            # Log success
            asset_count = len(data.get("assets", {}))
            self.logger.info(
                f"Loaded signal evaluations for {asset_count} assets from {self.filepath}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading signal evaluations: {e}")
            return False
    
    def save(self) -> bool:
        """Save signal evaluations to disk.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Update metadata
            self.evaluations["metadata"]["last_updated"] = datetime.now().isoformat()
            self.evaluations["metadata"]["num_assets"] = len(self.evaluations.get("assets", {}))
            
            # Count total evaluations
            eval_count = 0
            for asset_id, signals in self.evaluations.get("assets", {}).items():
                eval_count += len(signals)
            self.evaluations["metadata"]["num_evaluations"] = eval_count
            
            # Save to disk
            with open(self.filepath, 'w') as f:
                json.dump(self.evaluations, f, indent=2)
            
            self.logger.info(
                f"Saved {eval_count} signal evaluations for "
                f"{len(self.evaluations.get('assets', {}))} assets to {self.filepath}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal evaluations: {e}")
            return False
    
    def get_asset_evaluations(self, asset_id: str) -> Dict[str, Any]:
        """Get evaluations for a specific asset.
        
        Args:
            asset_id: ID of the asset to get evaluations for
            
        Returns:
            dict: Dictionary mapping signal names to evaluations for the asset
        """
        return self.evaluations.get("assets", {}).get(asset_id, {})
    
    def has_asset(self, asset_id: str) -> bool:
        """Check if an asset has stored evaluations.
        
        Args:
            asset_id: ID of the asset to check
            
        Returns:
            bool: True if the asset has evaluations, False otherwise
        """
        return asset_id in self.evaluations.get("assets", {})
    
    def has_signal_evaluation(self, asset_id: str, signal_name: str) -> bool:
        """Check if a specific signal has an evaluation for an asset.
        
        Args:
            asset_id: ID of the asset to check
            signal_name: Name of the signal to check
            
        Returns:
            bool: True if the signal has an evaluation for the asset, False otherwise
        """
        asset_evals = self.evaluations.get("assets", {}).get(asset_id, {})
        return signal_name in asset_evals
    
    def get_signal_evaluation(self, asset_id: str, signal_name: str) -> Optional[Dict[str, Any]]:
        """Get the evaluation for a specific signal and asset.
        
        Args:
            asset_id: ID of the asset
            signal_name: Name of the signal
            
        Returns:
            dict or None: Evaluation for the signal and asset, or None if not found
        """
        asset_evals = self.evaluations.get("assets", {}).get(asset_id, {})
        return asset_evals.get(signal_name)
    
    def add_signal_evaluation(self, asset_id: str, signal_name: str, evaluation: Dict[str, Any]) -> None:
        """Add a signal evaluation for an asset.
        
        Args:
            asset_id: ID of the asset
            signal_name: Name of the signal
            evaluation: Evaluation data for the signal
        """
        # Ensure the asset exists in evaluations
        if "assets" not in self.evaluations:
            self.evaluations["assets"] = {}
        
        if asset_id not in self.evaluations["assets"]:
            self.evaluations["assets"][asset_id] = {}
        
        # Add the evaluation timestamp if not present
        if "timestamp" not in evaluation:
            evaluation["timestamp"] = datetime.now().isoformat()
        
        # Add the evaluation
        self.evaluations["assets"][asset_id][signal_name] = evaluation
        self.logger.debug(f"Added evaluation for {signal_name} on {asset_id}")
    
    def remove_asset(self, asset_id: str) -> bool:
        """Remove all evaluations for an asset.
        
        Args:
            asset_id: ID of the asset to remove
            
        Returns:
            bool: True if the asset was removed, False if it wasn't found
        """
        if asset_id in self.evaluations.get("assets", {}):
            del self.evaluations["assets"][asset_id]
            self.logger.info(f"Removed all evaluations for {asset_id}")
            return True
        return False
    
    def get_all_assets(self) -> List[str]:
        """Get list of all assets with evaluations.
        
        Returns:
            list: List of asset IDs with evaluations
        """
        return list(self.evaluations.get("assets", {}).keys())
    
    def get_asset_effective_signals(self, asset_id: str) -> Dict[str, Any]:
        """Get all effective signals for an asset.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            dict: Dictionary mapping signal names to evaluations for effective signals
        """
        asset_evals = self.evaluations.get("assets", {}).get(asset_id, {})
        effective_signals = {}
        
        for signal_name, evaluation in asset_evals.items():
            if (evaluation.get("overall_effectiveness", False) 
                and evaluation.get("valid", False)
                and evaluation.get("optimal_holding_period", 0) > 0):
                effective_signals[signal_name] = evaluation
        
        return effective_signals
    
    def create_backup(self, suffix: Optional[str] = None) -> str:
        """Create a backup of the current evaluations.
        
        Args:
            suffix: Optional suffix for the filename (default: timestamp)
            
        Returns:
            str: Path to the backup file
        """
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_filename = f"{os.path.splitext(self.filename)[0]}_{suffix}.json"
        backup_path = os.path.join(self.storage_dir, backup_filename)
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(self.evaluations, f, indent=2)
            
            self.logger.info(f"Created signal evaluations backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return ""
    
    def _validate_loaded_data(self, data: Dict[str, Any]) -> bool:
        """Validate the structure of loaded data.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if the data is valid, False otherwise
        """
        # Check for required keys
        if "metadata" not in data or "assets" not in data:
            self.logger.error("Missing required keys in loaded data")
            return False
        
        # Check metadata
        metadata = data.get("metadata", {})
        if "version" not in metadata:
            self.logger.error("Missing version in metadata")
            return False
        
        # Validate version compatibility (basic check)
        version = metadata.get("version", "")
        major_version = version.split('.')[0] if '.' in version else version
        current_major = self.version.split('.')[0] if '.' in self.version else self.version
        
        if major_version != current_major:
            self.logger.warning(
                f"Version mismatch: loaded version {version}, "
                f"current version {self.version}"
            )
            # We continue anyway, but log the warning
        
        # Basic validation of assets structure
        assets = data.get("assets", {})
        if not isinstance(assets, dict):
            self.logger.error("Assets must be a dictionary")
            return False
        
        return True
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage.
        
        Returns:
            dict: Information about the storage
        """
        asset_count = len(self.evaluations.get("assets", {}))
        
        # Count signal evaluations and effective signals
        eval_count = 0
        effective_count = 0
        
        for asset_id, signals in self.evaluations.get("assets", {}).items():
            eval_count += len(signals)
            for signal_name, evaluation in signals.items():
                if evaluation.get("overall_effectiveness", False) and evaluation.get("valid", False):
                    effective_count += 1
        
        last_updated = self.evaluations.get("metadata", {}).get("last_updated", "Never")
        
        return {
            "filepath": self.filepath,
            "version": self.version,
            "last_updated": last_updated,
            "asset_count": asset_count,
            "evaluation_count": eval_count,
            "effective_count": effective_count
        } 