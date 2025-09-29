import pytest
from scripts.validate_config import check_agent_env_vars


def test_passing_config():
    """
    Happy Flow: correct config is passed
    """
    model_name = 'claude-sonnet-456098'
    api_keys = {
        'claude-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifja',
        'jentic-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifjakjhiuhuih-hiuhubjh',
    }

    model_temp = '0.2'
    try:
        check_agent_env_vars(model_name=model_name,
                         api_keys=api_keys,
                         model_temp=model_temp)
    except ValueError:
        assert False

    assert True

def test_failing_config_temp():
    """
    Tests if config check catches negative temperatures
    """
    model_name = 'claude-sonnet-456098'
    api_keys = {
        'claude-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifja',
        'jentic-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifjakjhiuhuih-hiuhubjh',
    }

    model_temp = '-0.2'
    try:
        check_agent_env_vars(model_name=model_name,
                             api_keys=api_keys,
                             model_temp=model_temp)
    except ValueError:
        assert True
        return

    assert False

def test_failing_missing_api_key_for_model():
    """
    Tests config check case where selected model's API KEY is not set
    """
    model_name = 'claude-sonnet-456098'
    api_keys = {
        'gpt-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifja',
        'jentic-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifjakjhiuhuih-hiuhubjh',
    }

    model_temp = '0.2'
    try:
        check_agent_env_vars(model_name=model_name,
                             api_keys=api_keys,
                             model_temp=model_temp)
    except ValueError:
        assert True
        return

    assert False

def test_failing_missing_jentic_key():
    """
    Tests env var check if jentic api key is not configured
    """
    model_name = 'claude-sonnet-456098'
    api_keys = {
        'claude-key': 'asdkeafoiejaoij1-0ei90ewe98uakjnekca njaieuhfaoeifja',
    }

    model_temp = '0.2'
    try:
        check_agent_env_vars(model_name=model_name,
                             api_keys=api_keys,
                             model_temp=model_temp)
    except ValueError:
        assert True
        return

    assert False

