import os
import dotenv

VALID_MODELS = (
    'gemini',
    'claude',
    'gpt',
    'vertex-ai'
)

ERROR_TEMPLATE = "!! {} environment variable for model: {} NOT SET !!"

def check_valid_model_type(model_name):
    for model in VALID_MODELS:
        if model in model_name:
            return True
    return False

def check_agent_env_vars(model_name: str, api_keys: dict, model_temp: str):
    if not model_name:
        raise ValueError("[ERROR] !! LLM_MODEL environment variable NOT SET !!")

    if not check_valid_model_type(model_name):
        raise ValueError(f"[ERROR] Invalid model name, please select one of {VALID_MODELS}")

    if model_name.lower().startswith(VALID_MODELS[0]) or model_name.lower().startswith(VALID_MODELS[3]):
        gemini_key = api_keys.get('gemini-key', None)

        if not gemini_key or gemini_key == "your-google-gemini-api-key-here":
            raise ValueError(ERROR_TEMPLATE.format("GEMINI_API_KEY", model_name))

    if model_name.lower().startswith(VALID_MODELS[1]):
        claude_key = api_keys.get('claude-key', None)

        if not claude_key or claude_key == "your-anthropic-api-key-here":
            raise ValueError(ERROR_TEMPLATE.format("ANTHROPIC_API_KEY", model_name))

    if model_name.lower().startswith(VALID_MODELS[2]):
        gpt_key = api_keys.get('gpt-key', None)

        if not gpt_key or gpt_key== "your-openapi-api-key-here":
            raise ValueError(ERROR_TEMPLATE.format("OPENAI_API_KEY", model_name))

    if model_temp and float(model_temp) < 0.0:
        raise ValueError("[ERROR] !! Invalid Model Temperature configured !!")

    jentic_api_key = api_keys.get('jentic-key', None)

    if not jentic_api_key or jentic_api_key == "your-anthropic-api-key-here":
        raise ValueError("[ERROR] !! JENTIC_AGENT_API_KEY environment variable NOT SET !!")

    print("Environment Variables are correct!")

if __name__=="__main__":
    print("...checking")
    dotenv.load_dotenv()
    model_name = os.getenv("LLM_MODEL", None)
    api_keys = {
        'gemini-key': os.getenv("GEMINI_API_KEY", None),
        'claude-key': os.getenv("ANTHROPIC_API_KEY", None),
        'gpt-key': os.getenv("OPENAI_API_KEY", None),
        'jentic-key': os.getenv("JENTIC_AGENT_API_KEY", None),
    }
    model_temp = os.getenv("LLM_TEMPERATURE", None)
    check_agent_env_vars(model_name=model_name,
                         api_keys=api_keys,
                         model_temp=model_temp)
    print("Finished Check")


