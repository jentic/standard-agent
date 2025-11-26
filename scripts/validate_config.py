 #!/usr/bin/env python3
  """
  scripts/validate_config.py

  Validate agent configuration files and environment before runtime.

  Features:
  - Validate JSON or YAML configuration files
  - Check required environment variables (user-specified, provider-agnostic)
  - Support agent-type–specific validation
  - Optional network connectivity checks with proper URL detection
  - CLI usage with helpful output

  Usage:
      python scripts/validate_config.py --config path/to/config.yaml
      python scripts/validate_config.py --config config.json --env-vars MY_API_KEY,OTHER_KEY
      python scripts/validate_config.py --config config.yaml --agent-type http_agent --check-network
  """

  from __future__ import annotations
  import argparse
  import json
  import os
  import re
  import socket
  import sys
  from pathlib import Path
  from typing import Any, Dict, List, Optional, Tuple
  from urllib.parse import urlparse

  # Optional YAML and JSON Schema support
  try:
      import yaml  # type: ignore
  except Exception:
      yaml = None  # type: ignore

  try:
      import jsonschema  # type: ignore
  except Exception:
      jsonschema = None  # type: ignore


  # -----------------------------
  # URL Validation Pattern
  # -----------------------------
  # Proper URL regex to avoid false positives (timestamps, log levels, etc.)
  URL_PATTERN = re.compile(
      r'^https?://'  # http:// or https:// required
      r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\.?|'  # domain
      r'localhost|'  # localhost
      r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP address
      r'(?::\d+)?'  # optional port
      r'(?:/?|[/?]\S*)?$',  # optional path
      re.IGNORECASE
  )


  def is_valid_url(value: str) -> bool:
      """Check if a string is a valid URL (not a timestamp, log level, etc.)."""
      if not isinstance(value, str):
          return False
      # Must match proper URL pattern with http/https scheme
      if URL_PATTERN.match(value):
          return True
      # Also validate using urlparse as secondary check
      try:
          parsed = urlparse(value)
          return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
      except Exception:
          return False


  # -----------------------------
  # Load Config (JSON/YAML)
  # -----------------------------
  def load_config(path: Path) -> Dict[str, Any]:
      if not path.exists():
          raise FileNotFoundError(f"Config file not found: {path}")

      text = path.read_text(encoding="utf-8")

      # Try JSON
      try:
          return json.loads(text)
      except Exception:
          pass

      # Try YAML
      if yaml is not None:
          try:
              return yaml.safe_load(text) or {}
          except Exception as e:
              raise ValueError(f"Failed to parse YAML: {e}") from e

      # Very basic fallback (not recommended)
      data: Dict[str, Any] = {}
      for line in text.splitlines():
          line = line.strip()
          if not line or line.startswith("#"):
              continue
          if ":" in line:
              key, val = line.split(":", 1)
              data[key.strip()] = val.strip().strip("\"'")

      if data:
          return data

      raise ValueError("Unable to parse config file. Install PyYAML for proper YAML support.")


  # -----------------------------
  # Environment Variable Checks
  # -----------------------------
  def check_env_vars(names: List[str], verbose: bool = False) -> Tuple[bool, List[str]]:
      """
      Check for required environment variables.

      Provider-agnostic: users specify their own required variables via CLI.
      No hardcoded provider-specific keys to maintain litellm compatibility.
      """
      missing = []
      for name in names:
          if not os.environ.get(name):
              missing.append(name)
              if verbose:
                  print(f"  [MISSING] {name}")
          elif verbose:
              print(f"  [OK] {name}")

      return (len(missing) == 0, missing)


  # -----------------------------
  # Schema Checks (Lightweight)
  # -----------------------------
  def basic_validate_config_schema(
      cfg: Dict[str, Any],
      agent_type: Optional[str] = None,
      verbose: bool = False
  ) -> Tuple[List[str], List[str]]:
      """
      Validate configuration schema.

      Returns tuple of (errors, warnings).

      Note: standard-agent's config.json is primarily for logging configuration.
      Agent-type specific validation is opt-in via --agent-type CLI flag.
      """
      errors: List[str] = []
      warnings: List[str] = []

      # Validate logging configuration if present (standard-agent's primary config use)
      if "logging" in cfg:
          logging_cfg = cfg["logging"]
          if isinstance(logging_cfg, dict):
              level = logging_cfg.get("level", "").upper()
              valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
              if level and level not in valid_levels:
                  warnings.append(
                      f"Invalid logging level '{level}'. Valid options: {', '.join(valid_levels)}"
                  )
              if verbose and level in valid_levels:
                  print(f"  [OK] Logging level: {level}")

      # Agent-type–specific logic (only when explicitly specified via CLI)
      at = agent_type or cfg.get("agent_type")

      if at:
          if verbose:
              print(f"  Validating for agent_type: {at}")

          if at == "discord_bot":
              if "discord" not in cfg:
                  errors.append("agent_type 'discord_bot' requires a 'discord' section.")
              else:
                  if "token" not in cfg["discord"] and "DISCORD_TOKEN" not in os.environ:
                      errors.append("Discord bot requires 'discord.token' or DISCORD_TOKEN env var.")

          elif at == "http_agent":
              if "endpoint" not in cfg:
                  errors.append("agent_type 'http_agent' requires an 'endpoint' key.")
              elif verbose:
                  print(f"  [OK] Found endpoint: {cfg['endpoint']}")

          # Extensible: add other agent types here
          else:
              if verbose:
                  print(f"  [INFO] No specific validation rules for agent_type '{at}'")
      else:
          # No agent_type specified - this is informational, not an error
          if verbose:
              print("  [INFO] No agent_type specified. Skipping agent-specific validation.")

      return errors, warnings


  # -----------------------------
  # URL Discovery (Fixed)
  # -----------------------------
  def discover_urls(obj: Any) -> List[str]:
      """
      Discover valid URLs in configuration.

      Uses proper URL validation to avoid false positives from:
      - Timestamps (e.g., "2024-01-01:12:00:00")
      - Log levels (e.g., "level:INFO")
      - Other colon-separated values
      """
      urls: List[str] = []

      def _extract(item: Any) -> None:
          if isinstance(item, dict):
              for v in item.values():
                  _extract(v)
          elif isinstance(item, list):
              for v in item:
                  _extract(v)
          elif isinstance(item, str) and is_valid_url(item):
              urls.append(item)

      _extract(obj)
      return urls


  # -----------------------------
  # Network Check
  # -----------------------------
  def try_connect_host(url_or_host: str, timeout: float = 3.0) -> Tuple[bool, str]:
      """Attempt TCP connection to validate network connectivity."""
      # Extract host:port from URL
      if "://" in url_or_host:
          try:
              parsed = urlparse(url_or_host)
              host = parsed.hostname or ""
              port = parsed.port or (443 if parsed.scheme == "https" else 80)
          except Exception:
              host = url_or_host.split("://", 1)[1].split("/", 1)[0]
              port = 443
      else:
          host = url_or_host
          port = 443

      if not host:
          return False, f"Could not extract host from: {url_or_host}"

      # Handle host:port format without scheme
      if ":" in host and not host.startswith("["):  # Not IPv6
          parts = host.rsplit(":", 1)
          host = parts[0]
          try:
              port = int(parts[1])
          except ValueError:
              pass

      try:
          addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
      except Exception as e:
          return False, f"DNS/resolve error for {host}:{port} — {e}"

      for fam, socktype, proto, canonname, sa in addr_info:
          try:
              s = socket.socket(fam, socktype, proto)
              s.settimeout(timeout)
              s.connect(sa)
              s.close()
              return True, f"Connected to {host}:{port}"
          except Exception:
              continue

      return False, f"Failed to connect to {host}:{port}."


  # -----------------------------
  # CLI Entry Point
  # -----------------------------
  def main(argv: Optional[List[str]] = None) -> int:
      parser = argparse.ArgumentParser(
          description="Validate agent configuration and environment.",
          formatter_class=argparse.RawDescriptionHelpFormatter,
          epilog="""
  Examples:
    %(prog)s --config config.yaml
    %(prog)s --config config.json --env-vars JENTIC_AGENT_API_KEY
    %(prog)s --config config.yaml --env-vars MY_KEY,OTHER_KEY --verbose
    %(prog)s --config config.yaml --agent-type http_agent --check-network
          """
      )

      parser.add_argument("--config", "-c", type=Path, required=True,
                          help="Path to JSON/YAML config file.")
      parser.add_argument("--env-vars", "-e", type=str,
                          help="Comma-separated list of environment vars to check. "
                               "If not specified, only checks JENTIC_AGENT_API_KEY.")
      parser.add_argument("--agent-type", "-t", type=str,
                          help="Agent type for specific validation (e.g., discord_bot, http_agent). "
                               "Overrides agent_type in config if specified.")
      parser.add_argument("--check-network", action="store_true",
                          help="Check network connectivity for URLs found in config.")
      parser.add_argument("--verbose", "-v", action="count", default=0,
                          help="Increase verbosity.")

      args = parser.parse_args(argv)

      problems: List[str] = []
      warnings: List[str] = []
      exit_code = 0
      verbose = args.verbose > 0

      if verbose:
          print(f"Validating config: {args.config}\n")

      # Load config
      try:
          cfg = load_config(args.config)
          if verbose:
              print("[OK] Config file loaded successfully.\n")
      except Exception as e:
          print(f"ERROR: Unable to load config: {e}", file=sys.stderr)
          return 2

      # Environment variable validation (provider-agnostic)
      # Only check user-specified vars, or minimal default
      if args.env_vars:
          env_list = [x.strip() for x in args.env_vars.split(",") if x.strip()]
      else:
          # Default: only check the generic Jentic key, not provider-specific keys
          # This maintains standard-agent's provider-agnostic philosophy with litellm
          env_list = ["JENTIC_AGENT_API_KEY"]

      if verbose:
          print("Checking environment variables:")

      ok, missing = check_env_vars(env_list, verbose=verbose)
      if not ok:
          problems.append(f"Missing environment variables: {', '.join(missing)}")
          exit_code = max(exit_code, 3)

      if verbose:
          print()

      # Schema validation
      if verbose:
          print("Validating configuration schema:")

      schema_errors, schema_warnings = basic_validate_config_schema(
          cfg, args.agent_type, verbose=verbose
      )

      if schema_errors:
          problems.extend(schema_errors)
          exit_code = max(exit_code, 4)

      if schema_warnings:
          warnings.extend(schema_warnings)

      if verbose:
          print()

      # Optional jsonschema validation
      if jsonschema is not None and isinstance(cfg, dict) and "$schema" in cfg:
          if verbose:
              print("[INFO] jsonschema detected — strict validation available.\n")

      # Network checks (with fixed URL discovery)
      if args.check_network:
          if verbose:
              print("Checking network connectivity:")

          endpoints = discover_urls(cfg)

          if not endpoints:
              if verbose:
                  print("  [INFO] No valid URLs found in config.\n")
          else:
              for ep in endpoints:
                  ok, msg = try_connect_host(ep)
                  if not ok:
                      problems.append(f"Network check failed for {ep}: {msg}")
                      exit_code = max(exit_code, 6)
                  elif verbose:
                      print(f"  [OK] {msg}")

              if verbose:
                  print()

      # Final Output
      if warnings:
          print("\nWarnings:", file=sys.stderr)
          for w in warnings:
              print(f"  - {w}", file=sys.stderr)

      if problems:
          print("\nValidation FAILED:\n", file=sys.stderr)
          for p in problems:
              print(f"  - {p}", file=sys.stderr)
          print(f"\nExit code: {exit_code}", file=sys.stderr)
      else:
          print("\nValidation PASSED.")
          if verbose:
              print("Configuration is valid.")

      return exit_code


  if __name__ == "__main__":
      sys.exit(main())
