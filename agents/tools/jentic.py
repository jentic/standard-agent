"""
Thin wrapper around jentic-sdk for centralized auth, retries, and logging.
"""
import asyncio
import os
import json
from typing import Any, Dict, List, Optional

from jentic import Jentic
from jentic.models import ApiCapabilitySearchRequest
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.exceptions import ToolNotFoundError, ToolExecutionError

# Use structlog for consistent logging
from utils.logger import get_logger
logger = get_logger(__name__)


class JenticTool(ToolBase):
    """Jentic-specific tool implementation with internal jentic metadata."""

    def __init__(self, schema: Dict[str, Any] | None = None):
        """
        Initialize JenticTool from jentic API results.

        Args:
            result: Raw result from jentic search or load API
        """
        # Initialize from search result
        if schema is None:
            schema = {}
        self._schema = schema
        self.tool_id = schema.get('workflow_id') or schema.get('operation_uuid') or schema.get('id') or ""
        super().__init__(self.tool_id)

        self.name = schema.get('summary', 'Unnamed Tool')
        self.description = schema.get('description', '') or f"{schema.get('method')} {schema.get('path')}"
        self.type = "workflow" if 'workflow_uuid' in schema else "operation"
        self.api_name = schema.get('api_name', 'unknown')
        self.method = schema.get('method')  # For operations
        self.path = schema.get('path')      # For operations
        self.required = schema.get('inputs', {}).get('required', []),
        self._parameters = schema.get('inputs', {}).get('properties', None)

    def __str__(self) -> str:
        """Short string description for logging purposes."""
        return f"JenticTool({self.id}, {self.name})"

    def get_summary(self) -> str:
        """Return summary information for LLM tool selection."""
        # Create description, preferring explicit description over method/path
        description = self.description
        if not description and self.method and self.path:
            description = f"{self.method} {self.path}"
        return f"{self.id}: {self.name} - {description} (API: {self.api_name})"

    def get_details(self) -> str:
        return json.dumps(self._schema, indent=4)

    def get_parameters(self) -> Dict[str, Any]:
        """Return detailed parameter schema for LLM parameter generation."""
        return self._parameters



class JenticClient(JustInTimeToolingBase):
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
        """
        self.api_key = api_key or os.getenv("JENTIC_API_KEY")
        self._jentic = Jentic(api_key=self.api_key)

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        """
        Search for workflows and operations matching a query.
        """
        logger.info("tool_search", query=query, top_k=top_k)

        # Call jentic search API directly
        results = asyncio.run(
            self._jentic.search_api_capabilities(
                ApiCapabilitySearchRequest(capability_description=query, max_results=top_k)
            )
        ).model_dump(exclude_none=False)

        top_results = (results.get("operations", []) + results.get("workflows", []))[:top_k]
        return [JenticTool(result) for result in top_results]


    def load(self, tool: ToolBase) -> ToolBase:
        """
        Load the detailed definition for a specific tool.
        """
        if not isinstance(tool, JenticTool):
            raise ValueError(f"Expected JenticTool, got {type(tool)}")

        logger.debug("tool_load", tool_id=tool.id, tool_type=tool.type)

        # Call jentic load API directly
        results = asyncio.run(
            self._jentic.load_execution_info(
                workflow_uuids=[tool.id] if tool.type == "workflow" else [],
                operation_uuids=[tool.id] if tool.type == "operation" else [],
                api_name=tool.api_name,
            )
        )

        # Find a specific result matching the tool we are looking for
        result = (results.get('workflows', {}).get(tool.id) or
                  results.get('operations', {}).get(tool.id))
        if result is None:
            raise ToolNotFoundError("Requested tool could not be loaded", tool)
        return JenticTool(result)


    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool with given parameters.
        """
        if not isinstance(tool, JenticTool):
            raise ValueError(f"Expected JenticTool, got {type(tool)}")

        logger.info("tool_execute", tool_id=tool.id, tool_type=tool.type, param_count=len(parameters))

        try:
            # Call jentic execute API directly
            if tool.type == "workflow":
                result = asyncio.run(self._jentic.execute_workflow(tool.id, parameters))
            else:
                result = asyncio.run(self._jentic.execute_operation(tool.id, parameters))

            # The result object from the SDK has a 'status' and 'outputs'.
            # A failure in the underlying tool execution is not an exception, but a
            # result with a non-success status.
            if not result.success:
                raise ToolNotFoundError(str(result.error), tool)
            return result.output

        except Exception as exc:
            # Re-raise as a ToolExecutionError so the reasoner can reflect.
            raise ToolExecutionError(str(exc), tool) from exc
