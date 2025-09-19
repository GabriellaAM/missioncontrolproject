import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
import mlflow
import tempfile
from abc import ABC, abstractmethod


class MetaStrategy(BaseStrategy):
    """Base class for meta-model strategies that use primary model labels."""

    is_meta_model = True

    def __init__(self, name: str, asset: str, primary_run_id: str):
        super().__init__(name)
        self.asset = asset
        self.primary_run_id = primary_run_id
        self._primary_labels = None
        self._working_client = None  # Will be set if file-based tracking is needed
        self._direct_file_access = False  # Will be True if using direct file system access
        self._artifact_file_path = None  # Direct path to artifact file
        
        # Validate primary run exists
        self._validate_primary_run()

    def _validate_primary_run(self):
        """Validate that the primary run exists and has the required artifacts."""
        try:
            # Try with current MLflow client (could be SQLite or file-based)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self.primary_run_id)
            
            # Extract primary run metadata
            self.primary_start_date = run.data.params.get('start_date')
            self.primary_end_date = run.data.params.get('end_date')
            self.primary_asset = run.data.params.get('asset')
            self.primary_train_test_split = float(run.data.params.get('train_test_split'))
            self.primary_strategy_type = run.data.params.get('strategy_type')
            
            # Check if triple_barrier_labels.csv artifact exists
            artifacts = client.list_artifacts(self.primary_run_id)
            has_labels = any(artifact.path == 'triple_barrier_labels.csv' for artifact in artifacts)
            
            if not has_labels:
                raise ValueError(f"Primary run {self.primary_run_id} does not have triple_barrier_labels.csv artifact")
                
            print(f"✅ Primary run {self.primary_run_id} validated successfully")
            print(f"   Primary model period: {self.primary_start_date} to {self.primary_end_date}")
            print(f"   Primary model asset: {self.primary_asset}")
            
        except Exception as first_error:
            # If MLflow client fails, try direct file system access to artifacts and extract dates from labels
            try:
                import os
                import glob
                import pandas as pd
                
                # Look for artifacts directory containing triple_barrier_labels.csv
                current_dir = os.getcwd()
                artifact_candidates = [
                    os.path.join(current_dir, 'mlruns', '*', self.primary_run_id, 'artifacts', 'triple_barrier_labels.csv'),
                    os.path.join(os.path.dirname(current_dir), 'mlruns', '*', self.primary_run_id, 'artifacts', 'triple_barrier_labels.csv'),
                    f'/Users/valter.rebelo/MissionControl/mlruns/*//{self.primary_run_id}/artifacts/triple_barrier_labels.csv'
                ]
                
                for pattern in artifact_candidates:
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        artifact_file = matching_files[0]
                        
                        # Load labels to extract date range
                        labels_df = pd.read_csv(artifact_file, index_col='timestamp', parse_dates=True)
                        self.primary_start_date = labels_df.index.min().strftime('%Y-%m-%d')
                        self.primary_end_date = labels_df.index.max().strftime('%Y-%m-%d')
                        self.primary_asset = self.asset  # Use current asset as fallback
                        self.primary_train_test_split = 0.8  # Default fallback
                        self.primary_strategy_type = ''  # Default fallback
                        
                        self._artifact_file_path = artifact_file
                        self._direct_file_access = True
                        print(f"✅ Primary run {self.primary_run_id} validated successfully (direct file access: {artifact_file})")
                        print(f"   Primary model period (extracted): {self.primary_start_date} to {self.primary_end_date}")
                        print(f"   Primary model asset: {self.primary_asset}")
                        return
            
            except Exception as second_error:
                pass
            
            # If both fail, raise the original error
            raise ValueError(f"Cannot validate primary run {self.primary_run_id}. Tried MLflow client and direct file access. Original error: {first_error}")

    def load_primary_signals(self, expected_start_date: str = None, expected_end_date: str = None, warmup_days: int = 0) -> pd.DataFrame:
        """Load primary model signals (full period) for meta-model predictions.

        Args:
            expected_start_date: Expected start date for signals (including warmup)
            expected_end_date: Expected end date for signals
            warmup_days: Number of warmup days needed
        """
        try:
            # Use MLflow client to download primary_signals.csv artifact
            client = getattr(self, '_working_client', None)
            if client is None:
                # Import here to avoid circular imports and ensure MLflow is configured
                import sys, os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from config import setup_mlflow
                setup_mlflow()
                client = mlflow.tracking.MlflowClient()

            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    # First try to load the new primary_signals.csv artifact
                    artifact_path = client.download_artifacts(
                        self.primary_run_id,
                        'primary_signals.csv',
                        tmp_dir
                    )
                    signals_df = pd.read_csv(artifact_path, index_col='timestamp', parse_dates=True)
                    print(f"✅ Loaded primary signals from primary_signals.csv")
                except Exception:
                    # Fallback to extracting signals from triple_barrier_labels.csv
                    print(f"⚠️  primary_signals.csv not found, extracting signals from triple_barrier_labels.csv")
                    artifact_path = client.download_artifacts(
                        self.primary_run_id,
                        'triple_barrier_labels.csv',
                        tmp_dir
                    )
                    combined_df = pd.read_csv(artifact_path, index_col='timestamp', parse_dates=True)
                    signals_df = combined_df[['signal']].copy()

                # Validate signal column exists
                if 'signal' not in signals_df.columns:
                    raise ValueError(f"Primary signals missing 'signal' column")

                print(f"   Loaded {len(signals_df)} primary signals from run {self.primary_run_id}")
                print(f"   Date range: {signals_df.index.min()} to {signals_df.index.max()}")
                print(f"   Non-null signals: {signals_df['signal'].notna().sum()}")

                # Handle warmup period alignment if parameters provided
                if expected_start_date and warmup_days > 0:
                    expected_start = pd.to_datetime(expected_start_date)
                    signals_start = signals_df.index.min()

                    if signals_start > expected_start:
                        # Primary signals don't cover the full warmup period
                        warmup_gap_days = (signals_start - expected_start).days
                        print(f"   ⚠️  Primary signals missing {warmup_gap_days} days of warmup period")
                        print(f"   🔧 Padding warmup period with zeros from {expected_start_date} to {signals_start}")

                        # Create warmup padding with zero signals
                        warmup_dates = pd.date_range(start=expected_start, end=signals_start, freq='D', inclusive='left')
                        warmup_padding = pd.DataFrame(
                            index=warmup_dates,
                            data={'signal': 0}  # Zero signals during warmup
                        )
                        warmup_padding.index.name = 'timestamp'

                        # Prepend warmup padding to signals
                        signals_df = pd.concat([warmup_padding, signals_df])
                        print(f"   ✅ Added {len(warmup_padding)} warmup days with zero signals")

                # Validate expected date range coverage if provided
                if expected_start_date and expected_end_date:
                    expected_start = pd.to_datetime(expected_start_date)
                    expected_end = pd.to_datetime(expected_end_date)
                    actual_start = signals_df.index.min()
                    actual_end = signals_df.index.max()

                    if actual_start > expected_start:
                        print(f"   ⚠️  WARNING: Primary signals start later than expected")
                        print(f"      Expected: {expected_start}, Actual: {actual_start}")

                    if actual_end < expected_end:
                        print(f"   ⚠️  WARNING: Primary signals end earlier than expected")
                        print(f"      Expected: {expected_end}, Actual: {actual_end}")

                print(f"   Final signals shape: {signals_df.shape}")
                return signals_df

        except Exception as e:
            raise RuntimeError(f"Failed to load primary signals from run {self.primary_run_id}: {e}")
    
    def load_primary_labels(self, expected_start_date: str = None, expected_end_date: str = None, warmup_days: int = 0) -> pd.DataFrame:
        """Load label artifacts (training labels only) from primary model run.

        Args:
            expected_start_date: Expected start date for labels (including warmup)
            expected_end_date: Expected end date for labels
            warmup_days: Number of warmup days needed
        """
        if self._primary_labels is not None:
            return self._primary_labels

        try:
            if self._direct_file_access and self._artifact_file_path:
                # Load directly from file system
                labels_df = pd.read_csv(self._artifact_file_path, index_col='timestamp', parse_dates=True)
            else:
                # Use MLflow client - ensure we have a valid client
                client = getattr(self, '_working_client', None)
                if client is None:
                    # Import here to avoid circular imports and ensure MLflow is configured
                    import sys, os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    from config import setup_mlflow
                    setup_mlflow()
                    client = mlflow.tracking.MlflowClient()

                # Download triple_barrier_labels.csv artifact
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifact_path = client.download_artifacts(
                        self.primary_run_id,
                        'triple_barrier_labels.csv',
                        tmp_dir
                    )

                    # Load the labels
                    labels_df = pd.read_csv(artifact_path, index_col='timestamp', parse_dates=True)

            # Validate required columns (labels may have NaN for test period)
            if 'label' not in labels_df.columns:
                raise ValueError(f"Primary labels missing 'label' column")

            print(f"✅ Loaded primary labels from run {self.primary_run_id}")
            print(f"   Original date range: {labels_df.index.min()} to {labels_df.index.max()}")

            # Handle warmup period alignment if parameters provided
            if expected_start_date and warmup_days > 0:
                expected_start = pd.to_datetime(expected_start_date)
                labels_start = labels_df.index.min()

                if labels_start > expected_start:
                    # Primary labels don't cover the full warmup period
                    warmup_gap_days = (labels_start - expected_start).days
                    print(f"   ⚠️  Primary labels missing {warmup_gap_days} days of warmup period")
                    print(f"   🔧 Padding warmup period with NaN labels from {expected_start_date} to {labels_start}")

                    # Create warmup padding with NaN labels (no ground truth during warmup)
                    warmup_dates = pd.date_range(start=expected_start, end=labels_start, freq='D', inclusive='left')

                    # Create warmup padding with same structure as original labels
                    warmup_columns = {col: None for col in labels_df.columns}  # NaN for all label columns during warmup
                    warmup_padding = pd.DataFrame(
                        index=warmup_dates,
                        data=warmup_columns
                    )
                    warmup_padding.index.name = 'timestamp'

                    # Prepend warmup padding to labels
                    labels_df = pd.concat([warmup_padding, labels_df])
                    print(f"   ✅ Added {len(warmup_padding)} warmup days with NaN labels")

            # Validate expected date range coverage if provided
            if expected_start_date and expected_end_date:
                expected_start = pd.to_datetime(expected_start_date)
                expected_end = pd.to_datetime(expected_end_date)
                actual_start = labels_df.index.min()
                actual_end = labels_df.index.max()

                if actual_start > expected_start:
                    print(f"   ⚠️  WARNING: Primary labels start later than expected")
                    print(f"      Expected: {expected_start}, Actual: {actual_start}")

                if actual_end < expected_end:
                    print(f"   ⚠️  WARNING: Primary labels end earlier than expected")
                    print(f"      Expected: {expected_end}, Actual: {actual_end}")

            # Only keep rows with non-null labels (training period)
            training_labels = labels_df.dropna(subset=['label'])

            self._primary_labels = labels_df  # Keep full dataset for reference
            print(f"   Final labels shape: {labels_df.shape}")
            print(f"   Training labels (non-null): {len(training_labels)}")
            print(f"   Available columns: {list(labels_df.columns)}")

            return labels_df

        except Exception as e:
            raise RuntimeError(f"Failed to load primary labels from run {self.primary_run_id}: {e}")

    def get_input_schema(self):
        """Override to include meta-specific columns (signal and label)."""
        # Get base schema from parent
        schema = super().get_input_schema()
        
        # Add meta-specific columns
        from mlflow.types import DataType, ColSpec, Schema
        
        meta_columns = [
            ColSpec(DataType.long, "signal"),  # Primary model signal
            ColSpec(DataType.long, "label")    # Triple barrier label
        ]
        
        # Combine base schema columns with meta columns
        all_columns = list(schema.inputs) + meta_columns
        
        return Schema(all_columns)

    @property
    def strategy_type(self) -> str:
        """Meta-models inherit strategy type from primary model."""
        return getattr(self, 'primary_strategy_type', 'trend_following')
    
    @property
    def implementation_type(self) -> str:
        """Meta-models are a distinct implementation type."""
        return "meta_model"
    
    @property
    def strategy_basis(self) -> str:
        """Meta-models have meta_models as their basis."""
        return "meta_models"

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate meta-model signals.
        
        For meta-models, the input data will include:
        - All regular features (price, volume, indicators, etc.)
        - signal: Primary model's signal (-1, 0, 1)
        - label: Triple barrier label (-1, 0, 1) - only available during training
        
        Args:
            data: DataFrame with features + signal + label columns
            params: Strategy parameters
            
        Returns:
            DataFrame with at least a 'signal' column
        """
        pass

    @abstractmethod
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
        """
        Optimize meta-model parameters.
        
        Args:
            data: DataFrame with features + signal + label columns
            train_start: Training start date
            train_end: Training end date
            n_trials: Number of optimization trials
            
        Returns:
            Dict with optimization results
        """
        pass
    
