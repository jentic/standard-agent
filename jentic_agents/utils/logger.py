"""
Singleton logger implementation with configuration from config.toml.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any

from jentic_agents.utils.config import load_config

class LoggerSingleton:
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config: Dict[str, Any] = load_config()
        self._setup_logging()
        LoggerSingleton._initialized = True
    
    def _setup_logging(self) -> None:
        """Set up logging based on the loaded configuration."""

        logging_config = self.config.get('logging', {})
        console_enabled = logging_config.get('console_enabled', True)
        file_enabled = logging_config.get('file_enabled', True)
        
        if not console_enabled and not file_enabled:
            logging.disable(logging.CRITICAL)
            return

        # Clear any existing handlers first
        logging.getLogger().handlers.clear()
        
        # Set root logger level (use INFO as fallback if not specified)
        root_level = self.config.get('level', 'INFO').upper()
        logging.getLogger().setLevel(root_level)
        
        # Console Handler
        if console_enabled:
            console_handler = logging.StreamHandler()
            console_level = logging_config.get('level', 'INFO').upper()
            console_handler.setLevel(console_level)
            
            # Use colored formatter if specified in config
            if logging_config.get('colored', False):
                formatter = ColoredFormatter(logging_config.get('format', '%(name)s:%(levelname)s: %(message)s'))
            else:
                formatter = logging.Formatter(logging_config.get('format', '%(name)s:%(levelname)s: %(message)s'))
            
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
            
        # File Handler  
        if file_enabled:
            log_path = Path(logging_config.get('path', 'jentic_agents/logs/standard_agent.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            
            file_level = logging_config.get('level', 'DEBUG').upper()
            file_handler.setLevel(file_level)
            
            file_format = logging_config.get('format', '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
            file_handler.setFormatter(logging.Formatter(file_format))
            
            logging.getLogger().addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance."""
        return logging.getLogger(name)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the current logging configuration."""
        return self.config


class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds color to the log level."""
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m', 'ERROR': '\033[31m', 'CRITICAL': '\033[35m'}
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# Create a single global instance for easy access
_logger_instance = LoggerSingleton()

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger from the singleton."""
    return _logger_instance.get_logger(name)

def get_config() -> Dict[str, Any]:
    """Convenience function to get the logging config."""
    return _logger_instance.get_config() 