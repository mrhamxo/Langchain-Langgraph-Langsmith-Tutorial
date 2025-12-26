from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

# LLM Model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Tools - MCP Client
mcp_client = MultiServerMCPClient(
    {
        "arith":{
            "transport": "stdio",
            "command": "python3",
            "args": ["C:/Users/Hamza/Desktop/arith_server.py"]
        },
        "expense":{
            "transport": "streamable_http", # if this fail try "sse"
            "url": "http://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

# State 
class ChatState(TypedDict):    
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    
    # Tools - MCP Client
    tools = await mcp_client.get_tools()
    print("Available tools:", tools)
    llm_with_tools = model.bind_tools(tools)
    
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
    chatbot = await build_graph()

    result = await chatbot.ainvoke({"messages": [HumanMessage(content="add expense of 1500 USD for laptop purchase on 24 december 2025")]})
    
    print("\n" + "="*50)
    print("Final Response:")
    print("="*50)
    print(result['messages'][-1].content)
    
if __name__ == "__main__":
    asyncio.run(main())