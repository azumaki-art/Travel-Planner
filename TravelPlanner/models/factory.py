from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from utils.helper import get_api_key

# Environment variables accepted for the Gemini key, in priority order.
GEMINI_KEY_ENV_VARS = ("Gemini_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

# Environment variables accepted for the Google Maps key, in priority order.
GOOGLE_MAPS_KEY_ENV_VARS = ("MAP_API_KEY", "GOOGLE_MAPS_API_KEY", "GOOGLE_MAP_API_KEY")

# Environment variables accepted for the DeepSeek key, in priority order.
DEEPSEEK_KEY_ENV_VARS = ("DEEPSEEK_API_KEY", "DeepSeek_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def get_google_maps_api_key() -> str:
    """Resolve the Google Maps key used by the location/POI/route tools."""
    return get_api_key(*GOOGLE_MAPS_KEY_ENV_VARS, service="Google Maps")


class LLMFactory:
    @staticmethod
    def get_llm(model=None, temperature=None):
        """Create a chat model instance for the given model name and temperature."""
        if model is None or temperature is None:
            raise ValueError("Both 'model' and 'temperature' must be specified.")

        if model.startswith("gemini"):
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=get_api_key(*GEMINI_KEY_ENV_VARS, service="Google Gemini"),
            )

        if model.startswith("gpt"):
            return ChatOpenAI(model=model, temperature=temperature, streaming=True)

        if model.startswith("deepseek"):
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                streaming=True,
                base_url=DEEPSEEK_BASE_URL,
                api_key=get_api_key(*DEEPSEEK_KEY_ENV_VARS, service="DeepSeek"),
            )

        raise ValueError(
            f"Model {model} is not supported. Use a 'gemini-*' , 'deepseek-v*' model "
            f"(e.g. gemini-2.5-flash) or a 'gpt-*' model."
        )


"""
1. LLMFactory class
Creates and returns the language model instance matching the requested name.

2. get_llm static method
Parameters:
    model: model name, e.g. "gemini-2.5-flash" or "gpt-4o-mini".
    temperature: sampling temperature (0 = deterministic).

Routing:
    "gemini*" -> ChatGoogleGenerativeAI, authenticated with Gemini_API_KEY
                 (aliases: GEMINI_API_KEY, GOOGLE_API_KEY).
    "gpt*"    -> ChatOpenAI, authenticated with OPENAI_API_KEY.
    anything else raises ValueError.

Note: unlike ChatOpenAI, ChatGoogleGenerativeAI has no `streaming` flag. Token
streaming is obtained by calling `.astream()` in agents.py, which is what makes
Gradio's incremental output work.
"""
