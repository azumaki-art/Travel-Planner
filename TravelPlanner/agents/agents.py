import os

from states.state import PublicState
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import StructuredTool
from models.factory import LLMFactory
from langchain_core.messages import HumanMessage, ToolMessage
from utils.helper import get_current_local_datetime

# Upper bound on the history handed to the model per call. Raw tool output
# (attraction descriptions, route steps, geocode matches) dominates the prompt,
# and without a bound every model call inside a turn re-sends the whole
# conversation, which is what made long planning turns take many minutes.
# Override with HISTORY_CHAR_BUDGET; set it to 0 to disable trimming.
HISTORY_CHAR_BUDGET = int(os.getenv("HISTORY_CHAR_BUDGET", "60000"))


class Agent:
    def __init__(self, model_name: str, temperature: float, prompt_template: str, tools: list[StructuredTool]):
        """initialize the agent with the specified model, temperature, prompt template, and tools"""
        self.llm = LLMFactory.get_llm(model=model_name, temperature=temperature)
        if tools:
            self.llm = self.llm.bind_tools(tools)
        self.prompt_template = prompt_template


def _trim_history(messages: list) -> list:
    """Keep only the most recent messages that fit within HISTORY_CHAR_BUDGET.

    Older messages are dropped so a long turn stops re-sending an ever-growing
    transcript. The graph state itself is never modified, so the UI still shows
    the full conversation; only the window sent to the model is bounded.

    The window is never allowed to begin with a ToolMessage, which would leave a
    tool result without the assistant message that requested it.
    """
    if HISTORY_CHAR_BUDGET <= 0:
        return list(messages)

    kept = []
    total = 0
    for message in reversed(messages):
        size = len(str(message.content))
        if kept and total + size > HISTORY_CHAR_BUDGET:
            break
        kept.append(message)
        total += size
    kept.reverse()

    while kept and isinstance(kept[0], ToolMessage):
        kept.pop(0)
    return kept


def _build_prompt(prompt_template: str, state: PublicState) -> list:
    """Build the message list: system-style preamble plus the running conversation.

    The prompt template is sent as the first HumanMessage (some providers, e.g.
    Gemini, handle a leading system instruction differently), so the first user
    turn is embedded into it via `first_user_message`.
    """
    first_prompt = HumanMessage(prompt_template.format(
        current_time=get_current_local_datetime(),
        first_user_message=state['messages'][0].content
    ))
    return [first_prompt] + _trim_history(state['messages'][1:])


def _merge_chunks(chunks: list) -> AIMessageChunk:
    """Merge streamed chunks into one message.

    Merging preserves both the accumulated text and the tool calls that were
    streamed as `tool_call_chunks`, so `tools_condition` can still route to the
    ToolNode.
    """
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


## AsyncAgent
class AsyncAgent(Agent):
    async def __call__(self, state: PublicState):
        """Stream the model response and return it as a single message.

        `astream` (rather than `ainvoke`) is required so that LangChain emits
        `on_llm_new_token` events. webrun.py relies on those to render the reply
        incrementally inside Gradio; with `ainvoke` the whole answer would appear
        at once for providers such as Gemini.
        """
        prompt = _build_prompt(self.prompt_template, state)
        chunks = [chunk async for chunk in self.llm.astream(prompt)]
        if not chunks:
            return {'messages': []}
        return {'messages': [_merge_chunks(chunks)]}


## SyncAgent
class SyncAgent(Agent):
    def __call__(self, state: PublicState):
        prompt = _build_prompt(self.prompt_template, state)
        response = self.llm.invoke(prompt)
        return {'messages': [response]}


"""
---
## 1. __Agent Class__

The **Agent** class is a base class used to initialize a language model (LLM) and its associated tools.

__init__ method:

- __model_name__: Specifies the name of the language model to use.
- __temperature__: Controls the randomness of text generation; higher values produce more random output.
- __prompt_template__: A template used to construct the combined system prompt and user messages.
- __tools__: A list of `StructuredTool` objects that can be bound to the language model.

During initialization, the Agent class retrieves the specified language model through `LLMFactory.get_llm`, and binds any provided tools to the model. The `prompt_template` is stored as an instance variable for later use.

---

## 2. __AsyncAgent Class__

The __AsyncAgent__ class inherits from `Agent` and implements asynchronous LLM invocation.

__call__ method:

 __state__: A `PublicState` object containing the current state information, typically including the list of user messages.

The method builds `prompt` via `_build_prompt` and then consumes `self.llm.astream(prompt)`, merging the
streamed chunks into a single `AIMessageChunk`. Streaming (instead of `ainvoke`) keeps token-level events
flowing to the Gradio UI while still producing a complete message, including any tool calls.

---

## 3. __SyncAgent Class__

The __SyncAgent__ class also inherits from `Agent`, but performs synchronous LLM invocation.

__call__ method:

Builds the same prompt and calls `self.llm.invoke(prompt)`, returning the complete response.

---

4. __Other Dependencies__

- __PublicState__: A state-management class used to store and manage the current state information.
- __LLMFactory__: A factory class responsible for retrieving the appropriate language model based on the model name and temperature.
- __StructuredTool__: A tool definition class representing tools that can be bound to the language model.
- __HumanMessage__ and __ToolMessage__: Classes used to construct and pass messages.
- __get_current_local_datetime__: A helper function used to obtain the current local time.

"""
