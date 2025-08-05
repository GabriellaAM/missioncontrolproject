"""
ZenML Pipeline Runner - Handles pipeline execution properly
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from zenml.client import Client
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from ..pipelines.enhanced_validation_pipeline import (
    enhanced_asset_diagnostic_pipeline,
    model_comparison_pipeline,
    batch_enhanced_diagnostics
)


class PipelineRunner:
    """Handles ZenML pipeline execution with proper configuration"""
    
    def __init__(self):
        self.client = Client()
        self.logger = logging.getLogger(__name__)
        
    def run_diagnostic_pipeline(
        self,
        asset_id: str,
        primary_model_mode: str = 'ml_based',
        primary_model_config: Optional[Dict[str, Any]] = None,
        meta_model_type: str = 'random_forest',
        data_dir: str = 'data',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the enhanced diagnostic pipeline and return results"""
        
        try:
            # Configure pipeline
            pipeline_instance = enhanced_asset_diagnostic_pipeline.with_options(
                run_name=f"diagnostics_{asset_id}_{primary_model_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                enable_cache=False,  # Disable caching for fresh runs
                settings={
                    "docker": DockerSettings(
                        required_integrations=[MLFLOW],
                        skip_build=True  # Skip Docker build for local runs
                    )
                }
            )
            
            # Run pipeline
            self.logger.info(f"Starting ZenML pipeline for {asset_id} with {primary_model_mode} model")
            
            # Execute the pipeline
            pipeline_run = pipeline_instance(
                asset_id=asset_id,
                primary_model_mode=primary_model_mode,
                primary_model_config=primary_model_config,
                meta_model_type=meta_model_type,
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            
            # For local orchestrator, the pipeline runs synchronously and returns the result directly
            if hasattr(pipeline_run, 'get') and isinstance(pipeline_run, dict):
                return pipeline_run
            
            # For other orchestrators, we need to wait for completion and fetch results
            # This is a simplified version - in production you'd want more robust handling
            if hasattr(pipeline_run, 'get_status'):
                self.logger.info(f"Pipeline run ID: {pipeline_run.id}")
                # Wait for completion (simplified - in production use proper polling)
                import time
                max_wait = 300  # 5 minutes
                wait_time = 0
                while pipeline_run.get_status() == 'running' and wait_time < max_wait:
                    time.sleep(5)
                    wait_time += 5
                
                if pipeline_run.get_status() == 'completed':
                    # Get the output from the last step
                    last_step = list(pipeline_run.steps.values())[-1]
                    return last_step.outputs
                else:
                    raise RuntimeError(f"Pipeline failed with status: {pipeline_run.get_status()}")
            
            # If we get here, assume it's a direct return (local orchestrator)
            return pipeline_run
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise