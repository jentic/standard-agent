"""
Singleton logger implementation with configuration from a JSON file.
"""
import json
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any


class LoggerSingleton:
    """Singleton logger that reads its configuration from a JSON file."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config: Dict[str, Any] = self._load_config()
        self._setup_logging()
        LoggerSingleton._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON, with fallbacks."""
        config_path = Path(__file__).parent / "../../config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('logging', {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}  # Return empty if file not found or invalid
    
    def _setup_logging(self) -> None:
        """Set up logging based on the loaded configuration."""
        if not self.config.get('enabled', True):
            logging.disable(logging.CRITICAL)
            return

        # Set root logger level
        root_level = self.config.get('level', 'INFO').upper()
        logging.getLogger().setLevel(root_level)
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Console Handler
        console_cfg = self.config.get('console', {})
        if console_cfg.get('enabled', True):
            console_handler = logging.StreamHandler()
            console_level = console_cfg.get('level', root_level).upper()
            console_handler.setLevel(console_level)
            
            formatter_cls = ColoredFormatter if console_cfg.get('colored', True) else logging.Formatter
            console_format = console_cfg.get('format', '%(name)s:%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter_cls(console_format))
            
            logging.getLogger().addHandler(console_handler)
            
        # File Handler
        file_cfg = self.config.get('file', {})
        if file_cfg.get('enabled', False):
            log_path = Path(file_cfg.get('path', 'logs/actbots.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            rotation_cfg = file_cfg.get('rotation', {})
            if rotation_cfg.get('enabled', True):
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path,
                    maxBytes=rotation_cfg.get('max_bytes', 10485760),
                    backupCount=rotation_cfg.get('backup_count', 5)
                )
            else:
                file_handler = logging.FileHandler(log_path)
            
            file_level = file_cfg.get('level', 'DEBUG').upper()
            file_handler.setLevel(file_level)
            
            file_format = file_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # Use a standard formatter for the file to avoid color codes
            file_handler.setFormatter(logging.Formatter(file_format))
            
            logging.getLogger().addHandler(file_handler)
            
        # Configure specific loggers
        loggers_cfg = self.config.get('loggers', {})
        for name, cfg in loggers_cfg.items():
            if 'level' in cfg:
                logging.getLogger(name).setLevel(cfg['level'].upper())

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