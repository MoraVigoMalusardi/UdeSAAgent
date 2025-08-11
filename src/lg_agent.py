#hay que correrlo con: python -m LangGraphAgent.LG_Agent


from lg_state import graph_builder
from langgraph.graph import START, END
from lg_nodes import tools,  route_tools, planner_node, chatbot, BasicToolNode, is_complex_question
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import AIMessage, HumanMessage
import uuid

def create_graph():
    # ------------------------------------ GRAPH ------------------------------------
    # The first argument is the unique node name, The second argument is the function or object that will be called whenever the node is used.
    tool_node = BasicToolNode(tools=tools) # Create an instance of the BasicToolNode with the tools

    def entry_router(state: dict) -> str:
        last_user_msg = state["messages"][-1] 
        if last_user_msg and isinstance(last_user_msg, HumanMessage) and is_complex_question(last_user_msg.content):
            return "planner"
        return "chatbot"
    
    #1. ---- Set nodes ----
    graph_builder.add_node("planner", planner_node)

    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_node("tools", tool_node)


    #2. ---- Add edges ---- 
    graph_builder.add_conditional_edges(
        START,
        entry_router,
        {"planner": "planner", "chatbot": "chatbot"}
    )

    graph_builder.add_edge("planner", "chatbot")

    # The 'tools_condition' function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")

    #3. ---- Compile the graph ---- 
    graph = graph_builder.compile(checkpointer=memory)
    return graph


# ------------------------------------ RUN --------------------------------------

def get_thread_id(user_id: str) -> str:
    # Usa un namespace fijo, p. ej. NAMESPACE_DNS, para que siempre genere el mismo UUID v5
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(user_id)).hex

def stream_graph_updates(user_input: str, user_id: str) -> str:
    """
    Stream updates from the graph based on user input and return only the assistant’s content.
    Args:
        user_input (str): The input from the user to be processed by the graph.
        user_id (str): The ID of the user (used to track thread memory).
        top_k (int): Number of top documents to retrieve.
    Returns:
        str: The concatenated content of all assistant messages in the stream.
    """
    config = {
        "configurable": {
            "thread_id": get_thread_id(user_id)
            # "top_k": top_k  # Configurable parameter to set the number of top documents to retrieve
        }
    }

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    ai_response = ""
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if isinstance(msg, AIMessage):
                ai_response += msg.content
            msg.pretty_print()
    return ai_response


def main():
    """ Main function to run the chatbot and handle user input. """
    print("Buen día! Soy un agente de la Universidad de San Andres, ¿en qué puedo ayudarte hoy?")
    print("Escribí 'quit', 'exit' o 'q' para salir.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Un gusto haber hablado con vos!")
                break
            stream_graph_updates(user_input, user_id="default_user")
        except Exception as e:
            # Imprimimos cual fue el error
            print(f"Error: {e}")
            break

# GLOBAL VARIABLE 
memory = MemorySaver()
graph = create_graph()  
print(graph.get_graph().print_ascii())

def debug_stream(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for i, event in enumerate(events):
        # Imprime la forma cruda de cada evento
        print(f"=== EVENT {i} ===")
        print(repr(event))
        print()
    return "DEBUG: stream impreso en consola."

if __name__ == "__main__":
    main()
    #debug_stream("hola")