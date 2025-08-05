"""
Simple configuration loader for Soros system
"""
import yaml
import os
from pathlib import Path


class SorosConfig:
    """Simple configuration loader for file paths"""
    
    def __init__(self, config_path="soros_config.yaml"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to YAML config file (relative to project root)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        # Find project root (look for soros_config.yaml)
        current_dir = Path.cwd()
        project_root = current_dir
        
        # Look for config file in current directory and parent directories
        for parent in [current_dir] + list(current_dir.parents):
            config_file = parent / self.config_path
            if config_file.exists():
                project_root = parent
                break
        else:
            raise FileNotFoundError(f"Config file {self.config_path} not found in {current_dir} or parent directories")
        
        # Load YAML config
        with open(project_root / self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert relative paths to absolute paths
        for key, path in config['data'].items():
            if not os.path.isabs(path):
                config['data'][key] = str(project_root / path)
        
        return config
    
    def get_data_paths(self):
        """Get all data paths as a dictionary"""
        return self.config['data']
    
    def get_path(self, key):
        """Get a specific path by key"""
        return self.config['data'].get(key)


def load_config(config_path="soros_config.yaml"):
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        SorosConfig instance
    """
    return SorosConfig(config_path)