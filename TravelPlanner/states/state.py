from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class PublicState(TypedDict):
    messages: Annotated[list, add_messages]


"""
TypedDict:
    From the typing module, used to define dictionaries with specific fields and types.
    It allows you to explicitly specify the name and type of each field in a dictionary, similar to defining a class but more lightweight.
Annotated:
    From the typing module, used to attach additional metadata (annotations) to fields.
    It allows you to append other information on top of type annotations, which can be used by tools or frameworks.
add_messages:
    From the langgraph.graph.message module, is a function or utility used to handle message lists.

PublicState class
PublicState is a typed dictionary (TypedDict) used to represent a state object containing a list of messages.
Fields:
    messages:
        Type is list, representing a list of messages.
        The messages field is annotated using Annotated, attaching the add_messages function.
        This means the messages field is not just an ordinary list, but also carries additional metadata or behavior (provided by add_messages).
        For example, add_messages may be used to perform certain operations (such as validation, formatting, or logging) when adding messages to the messages list.
"""
