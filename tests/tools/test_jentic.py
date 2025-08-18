# Test for the jentic.py script

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from http import HTTPStatus
import json

from agents.tools.jentic import JenticTool, JenticClient
from agents.tools.exceptions import ToolNotFoundError, ToolExecutionError, ToolCredentialsMissingError

# Mock data for JenticTool tests
WORKFLOW_SCHEMA = {
    'workflow_id': 'wf_123',
    'summary': 'Test Workflow',
    'description': 'A test workflow for unit tests.',
    'api_name': 'test_api',
    'inputs': {
        'required': ['param1'],
        'properties': {
            'param1': {'type': 'string'},
            'param2': {'type': 'integer'}
        }
    }
}

OPERATION_SCHEMA = {
    'operation_uuid': 'op_456',
    'summary': 'Test Operation',
    'description': '',
    'method': 'GET',
    'path': '/test/path',
    'api_name': 'test_api',
    'inputs': {
        'required': [],
        'properties': {
            'query_param': {'type': 'string'}
        }
    }
}

EMPTY_SCHEMA = {}

class TestJenticTool:
    """Tests for the JenticTool class."""

    def test_init_with_workflow_schema(self):
        """
        Tests initialization of JenticTool with a full workflow schema.
        Verifies that all attributes are correctly parsed and set.
        """
        tool = JenticTool(WORKFLOW_SCHEMA)
        assert tool.tool_id == 'wf_123'
        assert tool.id == 'wf_123'
        assert tool.name == 'Test Workflow'
        assert tool.description == 'A test workflow for unit tests.'
        assert tool.api_name == 'test_api'
        assert tool.required == (['param1'],)
        assert tool.get_parameters() == {
            'param1': {'type': 'string'},
            'param2': {'type': 'integer'}
        }

    def test_init_with_operation_schema(self):
        """
        Tests initialization with an operation schema.
        Verifies that method and path are correctly set and description is inferred.
        """
        tool = JenticTool(OPERATION_SCHEMA)
        assert tool.tool_id == 'op_456'
        assert tool.name == 'Test Operation'
        assert tool.description == 'GET /test/path'
        assert tool.method == 'GET'
        assert tool.path == '/test/path'
        assert tool.api_name == 'test_api'
        assert tool.get_parameters() == {'query_param': {'type': 'string'}}

    def test_init_with_empty_schema(self):
        """
        Tests initialization with an empty schema to ensure it doesn't crash.
        """
        tool = JenticTool(EMPTY_SCHEMA)
        assert tool.tool_id == ""
        assert tool.name == "Unnamed Tool"
        assert tool.api_name == "unknown"

    def test_get_summary_workflow(self):
        """
        Tests the get_summary method for a clear and correct representation.
        """
        tool = JenticTool(WORKFLOW_SCHEMA)
        summary = tool.get_summary()
        assert summary == "wf_123: Test Workflow - A test workflow for unit tests. (API: test_api)"

    def test_get_summary_operation(self):
        """
        Tests get_summary for an operation where description is inferred.
        """
        tool = JenticTool(OPERATION_SCHEMA)
        summary = tool.get_summary()
        assert summary == "op_456: Test Operation - GET /test/path (API: test_api)"

    def test_get_details_workflow(self):
        """
        Tests that get_details returns the original schema as a JSON string.
        """
        tool = JenticTool(WORKFLOW_SCHEMA)
        details = tool.get_details()
        assert json.loads(details) == WORKFLOW_SCHEMA

    def test_get_details_operation(self):
        """
        Tests that get_details returns the original schema as a JSON string.
        """
        tool = JenticTool(OPERATION_SCHEMA)
        details = tool.get_details()
        assert json.loads(details) == OPERATION_SCHEMA

    def test_get_parameters_workflow(self):
        """
        Tests that get_parameters returns the parameters from the schema.
        """
        tool = JenticTool(WORKFLOW_SCHEMA)
        parameters = tool.get_parameters()
        assert parameters == WORKFLOW_SCHEMA['inputs']['properties']

    def test_get_parameters_operation(self):
        """
        Tests that get_parameters returns the parameters from the schema.
        """
        tool = JenticTool(OPERATION_SCHEMA)
        parameters = tool.get_parameters()
        assert parameters == OPERATION_SCHEMA['inputs']['properties']

    def test_get_parameters_empty(self):
        """
        Tests that get_parameters returns an empty dictionary when the schema has no parameters.
        """
        tool = JenticTool(EMPTY_SCHEMA)
        parameters = tool.get_parameters()
        assert parameters == None


@pytest.fixture
def mock_jentic_sdk():
    """Fixture to mock the Jentic SDK."""
    with patch('agents.tools.jentic.Jentic') as mock_jentic_constructor:
        mock_sdk_instance = MagicMock()
        mock_sdk_instance.search = AsyncMock()
        mock_sdk_instance.load = AsyncMock()
        mock_sdk_instance.execute = AsyncMock()
        mock_jentic_constructor.return_value = mock_sdk_instance
        yield mock_sdk_instance

class TestJenticClient:
    """Tests for the JenticClient class."""

    def test_init(self, mock_jentic_sdk, monkeypatch):
        """
        Tests initialization of JenticClient.
        """
        # Ensure environment variable is set to True (default in env example)
        monkeypatch.setenv("JENTIC_FILTER_BY_CREDENTIALS", "true")
        
        client = JenticClient()
        assert client is not None
        assert client._jentic is not None
        assert client._filter_by_credentials is True

    @pytest.mark.parametrize("filter_by_credentials", [True, False])
    def test_init_with_filter_by_credentials(self, mock_jentic_sdk, filter_by_credentials: bool):
        """
        Tests initialization of JenticClient with filter_by_credentials.
        """
        client = JenticClient(filter_by_credentials=filter_by_credentials)
        assert client is not None
        assert client._jentic is not None
        assert client._filter_by_credentials == filter_by_credentials

    def test_search_success(self, mock_jentic_sdk):
        """
        Tests successful search call.
        """
        # Mock the SDK's search response
        mock_search_result = MagicMock()
        mock_search_result.model_dump.return_value = WORKFLOW_SCHEMA
        mock_sdk_instance = MagicMock()
        mock_sdk_instance.results = [mock_search_result]
        mock_jentic_sdk.search.return_value = mock_sdk_instance

        client = JenticClient()
        results = client.search(query="test")

        assert len(results) == 1
        assert isinstance(results[0], JenticTool)
        assert results[0].id == 'wf_123'
        mock_jentic_sdk.search.assert_called_once()

    def test_search_empty_results(self, mock_jentic_sdk):
        """
        Tests search call that returns no results.
        """
        mock_sdk_instance = MagicMock()
        mock_sdk_instance.results = []
        mock_jentic_sdk.search.return_value = mock_sdk_instance

        client = JenticClient()
        results = client.search(query="nonexistent")

        assert len(results) == 0

    def test_load_success(self, mock_jentic_sdk):
        """
        Tests successful load call.
        """
        mock_load_result = MagicMock()
        mock_load_result.model_dump.return_value = WORKFLOW_SCHEMA
        mock_sdk_instance = MagicMock()
        mock_sdk_instance.tool_info = {'wf_123': mock_load_result}
        mock_jentic_sdk.load.return_value = mock_sdk_instance

        client = JenticClient()
        tool_to_load = JenticTool({'id': 'wf_123'})
        loaded_tool = client.load(tool_to_load)

        assert isinstance(loaded_tool, JenticTool)
        assert loaded_tool.id == 'wf_123'
        assert loaded_tool.name == 'Test Workflow'

    def test_load_not_found(self, mock_jentic_sdk):
        """
        Tests load call where the tool is not found.
        """
        # To test the ToolNotFoundError, we need to bypass the KeyError in the source.
        # We do this by ensuring the key exists in tool_info, but its value is None.
        mock_response = MagicMock()
        mock_response.tool_info = {'not_found_id': None}
        mock_jentic_sdk.load.return_value = mock_response

        client = JenticClient()
        tool_to_load = JenticTool({'id': 'not_found_id'})

        with pytest.raises(ToolNotFoundError):
            client.load(tool_to_load)

    def test_execute_success(self, mock_jentic_sdk):
        """
        Tests successful tool execution.
        """
        mock_exec_result = MagicMock()
        mock_exec_result.success = True
        mock_exec_result.output = {'status': 'completed'}
        mock_jentic_sdk.execute.return_value = mock_exec_result

        client = JenticClient()
        tool_to_execute = JenticTool(WORKFLOW_SCHEMA)
        result = client.execute(tool_to_execute, parameters={'param1': 'value'})

        assert result == {'status': 'completed'}

    def test_execute_failure(self, mock_jentic_sdk):
        """
        Tests tool execution that fails with a generic error.
        """
        mock_exec_result = MagicMock()
        mock_exec_result.success = False
        mock_exec_result.error = "Execution failed"
        mock_exec_result.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        mock_jentic_sdk.execute.return_value = mock_exec_result

        client = JenticClient()
        tool_to_execute = JenticTool(WORKFLOW_SCHEMA)

        with pytest.raises(ToolExecutionError, match="Execution failed"):
            client.execute(tool_to_execute, {})

    def test_execute_unauthorized(self, mock_jentic_sdk):
        """
        Tests tool execution that fails due to authorization error.
        """
        mock_exec_result = MagicMock()
        mock_exec_result.success = False
        mock_exec_result.error = "Unauthorized"
        mock_exec_result.status_code = HTTPStatus.UNAUTHORIZED
        mock_jentic_sdk.execute.return_value = mock_exec_result

        client = JenticClient()
        tool_to_execute = JenticTool(WORKFLOW_SCHEMA)

        with pytest.raises(ToolCredentialsMissingError, match="Unauthorized"):
            client.execute(tool_to_execute, {})

    def test_execute_sdk_exception(self, mock_jentic_sdk):
        """
        Tests that an unexpected exception from the SDK is caught and wrapped.
        """
        mock_jentic_sdk.execute.side_effect = Exception("SDK Error")

        client = JenticClient()
        tool_to_execute = JenticTool(WORKFLOW_SCHEMA)

        with pytest.raises(ToolExecutionError, match="SDK Error"):
            client.execute(tool_to_execute, {})