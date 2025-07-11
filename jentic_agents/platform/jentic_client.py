"""
Thin wrapper around jentic-sdk for centralized auth, retries, and logging.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from jentic_agents.tools.exceptions import ToolExecutionError

# Standard module logger
logger = logging.getLogger(__name__)


class JenticClient:
    """
    Centralized adapter over jentic-sdk that exposes search, load, and execute.
    This client is designed to work directly with live Jentic services and
    requires the Jentic SDK to be installed.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Jentic client.

        Args:
            api_key: Jentic API key. If None, reads from JENTIC_API_KEY environment variable.
        
        Raises:
            ImportError: If the 'jentic' SDK is not installed.
        """
        self.api_key = api_key or os.getenv("JENTIC_API_KEY")
        self._tool_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Lazily-import the official Jentic SDK.  We do this here (instead of at
        # module import-time) so unit-tests can monkey-patch the import path.
        try:
            from jentic import Jentic  # type: ignore
            self._sdk_client = Jentic(api_key=self.api_key)  # async SDK instance

            # Models are only needed inside methods, so we import them lazily to
            # avoid bloating start-up time.
            import importlib

            self._sdk_models = importlib.import_module("jentic.models")

            logger.info("JenticClient initialised with REAL Jentic services (async SDK detected).")

        except ImportError as exc:
            logger.error("The 'jentic' SDK could not be imported – ensure it is installed and available in the current environment.")
            raise ImportError(
                "The 'jentic' package is not installed or is an incompatible version. "
                "Please run 'pip install -U jentic'."
            ) from exc

        # Internal helper to synchronously call the async SDK methods.
        import asyncio

        def _sync(coro):
            """Run *coro* in a new event loop and return the result.

            If an event-loop is already running (e.g. inside a Jupyter notebook),
            fall back to `asyncio.get_event_loop().run_until_complete`.
            """
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Avoid deadlocks / unexpected behaviour in environments that
                # already have a running event loop (e.g. Jupyter). For now we
                # fail fast so the caller can decide how to integrate the async
                # SDK properly.
                raise RuntimeError(
                    "JenticClient synchronous wrapper called from within an "
                    "active asyncio event-loop. Use the async SDK directly or "
                    "await the underlying coroutine instead."
                )

            return asyncio.run(coro)

        self._sync = _sync

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for workflows and operations matching a query. Caches metadata for later use.
        """
        logger.info(f"Searching for tools matching query: '{query}' (top_k={top_k})")

        # Build request model for the SDK.
        RequestModel = getattr(self._sdk_models, "ApiCapabilitySearchRequest")

        search_request = RequestModel(capability_description=query, max_results=top_k)

        # Call the async SDK synchronously.
        results = self._sync(self._sdk_client.search_api_capabilities(search_request))

        # Pydantic model ➔ dict
        if hasattr(results, "model_dump"):
            results_dict = results.model_dump(exclude_none=False)
        else:
            # Fallback for non-Pydantic objects.
            results_dict = dict(results)

        return self._format_and_cache_search_results(results_dict, top_k)

    def _format_and_cache_search_results(self, payload: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Formats search results and caches tool metadata."""
        formatted_results = []
        
        # API returns e.g. {"workflows": [...], "operations": [...]}  – iterate over both.
        for tool_type in ("workflows", "operations"):
            for tool in payload.get(tool_type, []):
                tool_id = tool.get('workflow_id') or tool.get('operation_uuid')
                if not tool_id:
                    continue

                formatted_tool = {
                    "id": tool_id,
                    "name": tool.get('summary', 'Unnamed Tool'),
                    "description": tool.get('description') or f"{tool.get('method')} {tool.get('path')}",
                    "type": "workflow" if tool_type == 'workflows' else "operation",
                    "api_name": tool.get('api_name', 'unknown'),
                    "parameters": {}  # Loaded on demand by load()
                }
                formatted_results.append(formatted_tool)
                self._tool_metadata_cache[tool_id] = {
                    "type": formatted_tool["type"],
                    "api_name": formatted_tool["api_name"]
                }
        
        return formatted_results[:top_k]

    def load(self, tool_id: str) -> Dict[str, Any]:
        """
        Load the detailed definition for a specific tool by its ID.
        Uses cached metadata to determine if it's a workflow or operation.
        """
        logger.info(f"Loading tool definition for ID: {tool_id}")

        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")

        # Prepare and execute load request via SDK
        load_coro = self._sdk_client.load_execution_info(
            workflow_uuids=[tool_id] if tool_meta["type"] == "workflow" else [],
            operation_uuids=[tool_id] if tool_meta["type"] == "operation" else [],
            api_name=tool_meta["api_name"],
        )

        results = self._sync(load_coro)

        # Convert Pydantic → dict for downstream processing.
        if hasattr(results, "model_dump"):
            results = results.model_dump(exclude_none=False)

        return self._format_load_results(tool_id, results)


    def _format_load_results(self, tool_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Formats loaded tool definition into a consistent structure."""
        if 'workflows' in results and results['workflows']:
            workflow_key = list(results['workflows'].keys())[0]
            workflow = results['workflows'][workflow_key]
            return {
                "id": tool_id,
                "name": workflow['summary'],
                "description": workflow['description'],
                "type": "workflow",
                "parameters": workflow.get('inputs', {}).get('properties', {}),
                "executable": True,
            }
        elif 'operations' in results and results['operations']:
            operation = results['operations'][tool_id]
            return {
                "id": tool_id,
                "name": operation['summary'],
                "description": f"{operation['method']} {operation['path']}",
                "type": "operation",
                "parameters": operation.get('inputs', {}).get('properties', {}),
                "required": operation.get('inputs', {}).get('required', []),
                "executable": True,
            }
        raise ValueError(f"Could not format load result for tool_id: {tool_id}")

    def execute(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters. Uses cached metadata to determine execution type.
        """
        logger.info(f"Executing tool '{tool_id}' with params: {params}")
        
        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")
        
        try:
            if tool_meta["type"] == "workflow":
                exec_coro = self._sdk_client.execute_workflow(tool_id, params)
            else:
                exec_coro = self._sdk_client.execute_operation(tool_id, params)

            result = self._sync(exec_coro)

            # The result object from the SDK has a 'status' and 'outputs'.
            # A failure in the underlying tool execution is not an exception, but a
            # result with a non-success status.
            if hasattr(result, "status") and result.status != "success":
                error_payload = result.outputs if hasattr(result, "outputs") else str(result)
                logger.warning(
                    "Tool execution reported failure for tool '%s': %s", tool_id, error_payload
                )
                raise ToolExecutionError(f"Tool {tool_id} failed: {error_payload}")

            return {"status": "success", "result": result}

        except Exception as exc:
            logger.error("Jentic execution failed for tool '%s': %s", tool_id, exc)
            # Re-raise as a ToolExecutionError so the reasoner can reflect.
            raise ToolExecutionError(str(exc)) from exc
