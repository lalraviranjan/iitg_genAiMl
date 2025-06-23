import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

async def main():
    # Create a Streamable HTTP Transport to communicate with a locally running MCP server
    # This MCP server was started via FastMCP and is exposing tools (e.g., read_query, create_repo)
    transport = StreamableHttpTransport(
        url="http://127.0.0.1:6274/mcp",
        headers={"Accept": "application/json, text/event-stream"}
    )
    
    # Create a Client that talks to the MCP server over the above transport
    client = Client(transport)

    async with client:
        # Get the list of tools exposed by the MCP server
        tools = await client.list_tools()
        print("Found tools:", [t.name for t in tools])

        def make_async_tool(tool_name):
            async def wrapper(arg: str):
                # Try to parse the input as JSON. If it fails, use it as a plain string.
                parsed = None
                try:
                    parsed = json.loads(arg)
                except json.JSONDecodeError:
                    parsed = arg
                # Call the MCP tool with the parsed input
                return await client.call_tool(tool_name, parsed)
            
            # A sync fallback function for LangChain compatibility
            def sync_noop(arg: str):
                # Optional: return placeholder or raise NotImplemented if not needed
                raise RuntimeError("Sync call not supported—use the Agent.")

            return sync_noop, wrapper
        
        # Wrap each MCP tool into LangChain-compatible Tool objects
        lc_tools = []
        for t in tools:
            sync_fn, async_fn = make_async_tool(t.name)
            lc_tools.append(
                Tool(
                    name=t.name,
                    description=t.description or "",
                    func=sync_fn,
                    # Actual async function the agent will call
                    coroutine=async_fn
                )
            )

        # Initialize the LLM 
        llm = ChatOpenAI(temperature=0)
        
        # Initialize a LangChain agent with the tools, LLM, and custom system prompt
        # This agent will interpret user input, decide when and which tool to call,
        # and follow the ReAct pattern (Thought → Action → Action Input → Observation → Thought → ...)
        agent = initialize_agent(
            tools=lc_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": ("""
                    You are a tool-using AI assistant connected to two domains:
                    1. Sales DB Tools - for querying a SQLite sales database.
                    2. GitHub Tools - for creating repositories via GitHub API.

                    When a user request requires using a tool, always follow this format:
                    Thought: <Explain why the tool is needed>
                    Action: <Exact tool name, like 'read_query', 'list_tables', or 'create_repo'>
                    Action Input: <Tool-specific input - dictionary as required>

                    Examples
                   1. Sales DB Tools:
                    - To list tables in the sales database:
                    Thought: I need to see which tables exist in the Sales DB.
                    Action: list_tables
                    Action Input: {}

                    - To get sales for 10th June 2025:
                    Thought: I need to query the sales table for the date 10th June 2025.
                    Action: read_query
                    Action Input: {"query": "SELECT * FROM sales WHERE date = '2025-06-10'"}
                    
                    2. GitHub Tools
                    - To create a GitHub repo named 'my-repo':
                    Thought: I need to create a GitHub repository with the specified name.
                    Action: create_repo
                    Action Input: {"repo_name": "my-repo"}

                    """
                )
            },
            max_iterations=10
        )

        while True:
            user_input = input("\nAsk (or type 'exit'):\n> ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            try:
                response = await agent.arun(user_input)
                print("Response", response)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
