from langchain_core.tools import tool
from typing import Annotated, Tuple

@tool(response_format="content_and_artifact")
def save_info_and_clear_history(
    information_to_save: Annotated[str, "Useful information to save. Mainly the useful portion from the information returned by tools you called. Be as detailed as possible."],
) -> Tuple[str, str]:
    """Information saving tool. Saves the useful information currently obtained."""
    # In fact, as LangGraph tool does not have the ability to update graph state, we only return the tool message to the main agent.
    # The update of the graph state will be handled in the main agent logic.
    content = "Information saved."
    # The LLM only get the content, the information to save will be saved in the 'artifact' field of the tool message.
    return content, information_to_save

# test the tool
if __name__ == "__main__":
    print(save_info_and_clear_history.args_schema.model_json_schema())
    tool_call =     {
        "name": "save_info_and_clear_history",
        "args": {"information_to_save": "2026-03-14"},
        "id": "123",  # required
        "type": "tool_call",  # required
    }
    a=save_info_and_clear_history.invoke(tool_call)
    print(a)
