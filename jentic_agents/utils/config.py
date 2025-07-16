from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class LoggingConsole:
    enabled: bool
    colored: bool
    level: str
    format: str

@dataclasses.dataclass
class LoggingFile:
    enabled: bool
    level: str
    format: str
    path: str
    file_rotation: bool
    max_bytes: int
    backup_count: int

@dataclasses.dataclass
class LoggingLibraries:
    LiteLLM: str
    httpx: str
    httpcore: str

@dataclasses.dataclass
class Logging:
    console: LoggingConsole
    file: LoggingFile
    libraries: LoggingLibraries

@dataclasses.dataclass
class LLM:
    provider: str
    model: str

@dataclasses.dataclass
class Config:
    logging: Logging
    llm: LLM
