import os
import warnings

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_core._api.beta_decorator import LangChainBetaWarning

from graph.graph import init_app
from utils.helper import get_thread_id

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

_ = load_dotenv(find_dotenv())

# Any "gemini-*" model works; override with the MODEL_NAME environment variable.
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")

app = init_app(model_name=MODEL_NAME)


def _chunk_text(content) -> str:
    """Extract plain text from a streamed chunk.

    Providers normally stream a plain string, but Google models can return a list
    of content blocks, so both shapes are handled.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


async def process_message(user_message, chatbot_history, debug_history, thread_id):
    """Stream one user turn through the graph and yield the Gradio updates."""
    # Gradio's "messages" format is a list of {"role", "content"} dictionaries.
    history = list(chatbot_history or [])
    debug = debug_history or ""

    # Mint this session's thread id on its first message, then return it in the
    # outputs so Gradio stores it back into the session's State.
    if not thread_id:
        thread_id = get_thread_id()

    if not user_message or not user_message.strip():
        yield history, debug, thread_id
        return

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})

    formatted_user_message = HumanMessage(content=user_message)
    # The workflow can issue many tool calls in one turn (a coordinate lookup per
    # attraction, then a route per leg), so LangGraph's default recursion limit of
    # 25 is raised to keep a long planning turn from aborting.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    async for event in app.astream_events(
        {"messages": formatted_user_message}, config=config, version="v1"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            text = _chunk_text(event["data"]["chunk"].content)
            if text:
                history[-1]["content"] += text
                yield history, debug, thread_id

        elif kind == "on_tool_start":
            debug += f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}\n"
            yield history, debug, thread_id

        elif kind == "on_tool_end":
            debug += f"Done tool: {event['name']}\nTool output: {event['data'].get('output')}\n--\n"
            yield history, debug, thread_id

    # Never leave an empty assistant bubble behind (happens if a turn produced
    # only tool calls).
    if history and history[-1]["role"] == "assistant" and not history[-1]["content"]:
        history.pop()
    yield history, debug, thread_id


def clear_input():
    return ""


def start_gradio():
    with gr.Blocks(title="Travel Planner") as demo:
        gr.Markdown("# Travel Planner")

        # One unique LangGraph thread per browser session.
        #
        # NOTE: gr.State(get_thread_id()) does NOT work here. Gradio resolves a
        # callable initial value once, when the app is built (Component.__init__ ->
        # get_load_fn_and_initial_value -> value()), and then deep-copies that same
        # result into every session, so all sessions would share one thread.
        # Leaving the State empty and minting the id on the session's first message
        # keeps each session isolated.
        thread_state = gr.State()

        with gr.Row(equal_height=True) as chat_interface:
            chat_interface.elem_classes = ["full-height"]
            # Left column for debug info
            with gr.Column(scale=1):
                debug_info = gr.Textbox(
                    label="Debug Info",
                    lines=30,
                    interactive=False,
                    elem_id="debug-info"
                )

            # Right column for chat interface
            with gr.Column(scale=3):
                # Chatbot interface, using Gradio's OpenAI-style "messages" format
                chatbot = gr.Chatbot(
                    label="User-AI Chat",
                    show_label=False,
                    type="messages",
                    elem_id="chatbot"
                )

                # User input textbox
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here",
                    lines=3,
                    max_lines=5,
                    show_label=False,
                    elem_id="user-input"
                )

                submit_click = gr.Button("Send")  # Submit function for message input

        submit_inputs = [user_input, chatbot, debug_info, thread_state]
        submit_outputs = [chatbot, debug_info, thread_state]

        submit_click.click(
            process_message, submit_inputs, submit_outputs
        ).then(
            clear_input, None, user_input
        )

        user_input.submit(
            process_message, submit_inputs, submit_outputs
        ).then(
            clear_input, None, user_input
        )

    demo.launch()

# App entry point
if __name__ == "__main__":
    start_gradio()
