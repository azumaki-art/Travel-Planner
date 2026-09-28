"""Shared helpers for the travel planning robot."""
from datetime import datetime
import os
import uuid


def get_current_local_datetime() -> str:
    """Current local time, formatted for the agent prompt. Used in agents.py."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_thread_id() -> str:
    """Return a fresh, unique conversation/thread id.

    A UUID is used instead of a timestamp so that two sessions started within the
    same second can never share one LangGraph checkpointer thread.
    """
    return str(uuid.uuid4())


# Values that are clearly template placeholders rather than a real key.
_PLACEHOLDER_MARKERS = (
    "myapikey",
    "googlemap api key",
    "your_api_key",
    "your-api-key",
    "yourapikey",
    "changeme",
    "replace_me",
    "replaceme",
    "<your",
)


def get_api_key(*names: str, service: str = "") -> str:
    """Return the first non-empty environment variable among ``names``.

    Values are stripped of surrounding whitespace and quotes, because keys pasted
    into ``.env`` are commonly written as ``KEY='value'``.

    A descriptive RuntimeError is raised when the key is missing or is still an
    obvious placeholder, instead of letting a cryptic library error surface later.
    """
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        value = raw.strip().strip('"').strip("'").strip()
        if not value or value.lower() in _PLACEHOLDER_MARKERS:
            continue
        return value

    label = f" for {service}" if service else ""
    raise RuntimeError(
        f"Missing API key{label}. Please set one of [{', '.join(names)}] in the "
        f".env file next to webrun.py, then restart the app."
    )


"""
webrun.py (Gradio UI entry)
  ├── init_app(...)                → graph.graph builds the LangGraph workflow
  │     └── Agent / AsyncAgent     → agents.agents builds the LLM prompt
  │           └── prompt_template.format(current_time=..., first_user_message=...)
  │                 ▲
  │                 └ get_current_local_datetime()   ← helper.py
  │
  ├── get_thread_id()              ← helper.py → LangGraph config["configurable"]["thread_id"]
  │
  └── get_api_key(...)             ← helper.py → resolves Gemini / Google Maps credentials

Note: the previous `save_chat_history` helper was removed. It read
`state['chat_history']`, a key that PublicState never defines (only `messages`),
so it always raised KeyError and was never called by the UI.
"""
