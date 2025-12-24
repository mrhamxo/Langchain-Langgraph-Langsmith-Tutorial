from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

# LLM Model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Tools
# Calculator Tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


# Make tool list
tools = [calculator]
llm_with_tools = model.bind_tools(tools)

# State 
class ChatState(TypedDict):    
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph():
    
    async def chat_node(state: ChatState):
        message = state["messages"]
        response = await llm_with_tools.ainvoke(message)
        return {"messages": [response]}
    
    tool_node = ToolNode(tools)

    # defining Graph and node
    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")    
    graph.add_edge('chat_node', END)

    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke({"messages": [HumanMessage(content="find the module of 12345, 45 and give answer like a football commentator")]})
    
    print(result['messages'][-1].content)
    
if __name__ == "__main__":
    asyncio.run(main())