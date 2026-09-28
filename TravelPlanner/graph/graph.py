# Reference: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools
from langgraph.graph import StateGraph, START
from states.state import PublicState
from langgraph.prebuilt import ToolNode, tools_condition
from prompts.main import agent_prompt_template
from tools import *

def create_graph(model_name, is_async=True):
    graph = StateGraph(PublicState)
    tools = [web_search,
             get_location_coordinate,
             get_attractions_information,
             route_planning,
             search_nearby_poi,
             save_info_and_clear_history,
            ]

    if is_async:
        from agents.agents import AsyncAgent as MainAgent
    else:
        from agents.agents import SyncAgent as MainAgent

    travel_agent = MainAgent(
        model_name=model_name,
        temperature=0,
        prompt_template=agent_prompt_template,
        tools=tools)
    # Pass the "__call__" function in the Agent class to add_node. This function will be called when the node is invoked.
    # The function should be able to use the agent's 'llm' and 'prompt_template' attributes as they have been initialized
    # when the agent instance was created.
    graph.add_node("agent", travel_agent)

    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # Will either direct to a specific tool in tools or to the END node
    graph.add_edge("tools", "agent")
    return graph

def init_app(model_name, is_async=True):
    graph = create_graph(model_name, is_async)
    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    app = graph.compile(checkpointer=memory)
    return app



"""
Code Structure Explanation:
1. State graph workflow:
   Agent -> Conditional check -> Tool execution (if needed) -> Return to Agent -> ... loops until finished

2. Key nodes:
   - agent: responsible for decision-making and generating responses
   - tools: responsible for executing specific tool operations

3. State management:
   - Uses PublicState to pass state between nodes
   - State persistence is implemented via InMemorySaver

Notes:
1. Path dependencies:
   - Ensure that module paths such as agents.agents and states.state are correct
   - Run the app from the TravelPlanner directory, which is the import root

2. Async mode:
   - webrun.py uses is_async=True, because the Gradio UI drives the graph with
     `astream_events` and the AsyncAgent streams tokens from the model
   - Setting is_async=False selects SyncAgent for scripts/tests that prefer `.invoke`

3. Tool calls:
   - Tool functions must comply with langgraph's ToolNode calling conventions
   - Each tool should have clear input/output type definitions

4. Prompt engineering:
   - agent_prompt_template needs to be properly designed to guide the agent to use tools correctly
   - It is recommended to include tool usage instructions and format requirements in the prompt

"""
