"""
MLflow Configuration for SOROS Lab Experiments
"""

import mlflow
from pathlib import Path

def setup_mlflow():
    """Configure MLflow with SQLite backend - call this at the start of each experiment script"""
    
    # Keep database in experiments directory
    experiments_dir = Path(__file__).parent
    db_path = experiments_dir / "mlflow_experiments.db"
    tracking_uri = f"sqlite:///{db_path.absolute()}"
    
    # Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✅ MLflow configured with database: {db_path}")
    print(f"🌐 Start UI with: mlflow ui --backend-store-uri {tracking_uri}")
    
    return tracking_uri

def create_experiment(experiment_name):
    """Create a new experiment if it doesn't exist"""
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            print(f"✅ Using existing experiment: {experiment_name}")
            return mlflow.get_experiment_by_name(experiment_name).experiment_id
        else:
            raise e