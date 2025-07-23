from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class LLM:
    model: str

@dataclasses.dataclass
class LoggingConsole:
    enabled: bool
    colour_enabled: bool
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
class Config:
    llm: LLM
    logging: Logging