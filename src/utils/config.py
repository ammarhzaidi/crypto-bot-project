# Open src/utils/config.py and add this content:
import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the trading application."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self.config_data = {}
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get(self, section: str, key: str = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self.config_data.get(section, {})

        return self.config_data.get(section, {}).get(key)