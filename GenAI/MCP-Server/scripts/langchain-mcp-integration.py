import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from fastmcp.client import Client
from langchain.agents import AgentType
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

async def main():
    transport = StreamableHttpTransport(
        url="http://127.0.0.1:6274/mcp",
        headers={"Accept": "application/json, text/event-stream"}
    )
    client = Client(transport)

    async with client:
        tools = await client.list_tools()
        print("✅ Found tools:", [t.name for t in tools])

        def make_async_tool(tool_name):
            async def wrapper(arg: str):
                return await client.call_tool(tool_name, arg)
            return wrapper

        lc_tools = [
            Tool(name=t.name, func=make_async_tool(t.name), description=t.description or "")
            for t in tools
        ]

        llm = ChatOpenAI(temperature=0)
        agent = initialize_agent(
            tools=lc_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": (
                    "You are an assistant that can use tools.\n"
                    "Whenever you use a tool, you MUST reply with:\n"
                    "Thought: your reasoning\n"
                    "Action: tool_name\n"
                    "Action Input: input_for_tool"
                )
            },
            max_iterations=10  # 🔼 Increase this if needed
        )


        print("\n🧪 Max sale_amount for Widget A")
        print("➡️", agent.run("What is the maximum sale_amount for Widget B?"))

        # print("\n🧪 List all tables")
        # print("➡️", agent.run("List all tables in the database."))

if __name__ == "__main__":
    asyncio.run(main())
