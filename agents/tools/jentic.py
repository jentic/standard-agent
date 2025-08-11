"""
Thin wrapper around jentic-sdk for centralized auth, retries, and logging.
"""
import asyncio
import os
import json
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from jentic import Jentic
from jentic.lib.models import SearchRequest, LoadRequest, ExecutionRequest
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.exceptions import ToolError, ToolNotFoundError, ToolExecutionError, ToolCredentialsMissingError

from utils.logger import get_logger
logger = get_logger(__name__)


class JenticTool(ToolBase):
    """Jentic-specific tool implementation with internal jentic metadata."""

    def __init__(self, schema: Dict[str, Any] | None = None):
        """
        Initialize JenticTool from jentic API results.

        Args:
            schema: Raw result from jentic search or load API as plain dict
        """
        # Initialize from search result
        if schema is None:
            schema = {}
        self._schema = schema

        self.tool_id = schema.get('workflow_id') or schema.get('operation_uuid') or schema.get('id') or ""
        super().__init__(self.tool_id)

        self.name = schema.get('summary', 'Unnamed Tool')
        self.description = schema.get('description', '') or f"{schema.get('method')} {schema.get('path')}"
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

    def __init__(self, *, filter_by_credentials: Optional[bool] = None):
        """
        Initialize Jentic client.
        """
        self._jentic = Jentic()
        if filter_by_credentials is None:
            filter_by_credentials_env_val = os.getenv("JENTIC_FILTER_BY_CREDENTIALS", "false").strip().lower()
            filter_by_credentials = filter_by_credentials_env_val == "true"
        self._filter_by_credentials = bool(filter_by_credentials)

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        """
        Search for workflows and operations matching a query.
        """
        logger.info("tool_search", query=query, top_k=top_k, filter_by_credentials=self._filter_by_credentials)

        response = asyncio.run(self._jentic.search(SearchRequest(query=query, limit=top_k, filter_by_credentials=self._filter_by_credentials,)))
        return [JenticTool(result.model_dump(exclude_none=False)) for result in response.results] if response.results else []


    def load(self, tool: ToolBase) -> ToolBase:
        """
        Load the detailed definition for a specific tool.
        """
        if not isinstance(tool, JenticTool):
            raise ValueError(f"Expected JenticTool, got {type(tool)}")

        logger.debug("tool_load", tool_id=tool.id)

        # Call jentic load API directly
        response = asyncio.run(self._jentic.load(LoadRequest(ids=[tool.id])))

        # Find a specific result matching the tool we are looking for
        result = response.tool_info[tool.id]
        if result is None:
            raise ToolNotFoundError("Requested tool could not be loaded", tool)
        return JenticTool(result.model_dump(exclude_none=False))


    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool with given parameters.
        """
        if not isinstance(tool, JenticTool):
            raise ValueError(f"Expected JenticTool, got {type(tool)}")

        logger.info("tool_execute", tool_id=tool.id, param_count=len(parameters))

        try:
            # Call jentic execute API directly
            result = asyncio.run(self._jentic.execute(ExecutionRequest(id=tool.id, inputs=parameters)))

            # The result object from the SDK has a 'status' and 'outputs'.
            # A failure in the underlying tool execution is not an exception, but a
            # result with a non-success status.
            if not result.success:
                if result.status_code == HTTPStatus.UNAUTHORIZED:
                    raise ToolCredentialsMissingError(result.error, tool)

                raise ToolExecutionError(result.error, tool)
            return result.output

        except ToolError:
            raise
        except Exception as exc:
            # Normalize any unexpected error as ToolExecutionError so the reasoner can handle it.
            raise ToolExecutionError(str(exc), tool) from exc
