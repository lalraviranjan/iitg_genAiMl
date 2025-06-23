################### run the server - python sqlite_mcp_server.py ############################
from fastmcp import FastMCP
import sqlite3, os
from github import Github
import requests

# Load GitHub credentials
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

# Compute DB path
script_dir = os.path.dirname(os.path.abspath(__file__))
sales_db_path = os.path.normpath(os.path.join(script_dir, '..', 'database', 'sales.db'))
assert os.path.exists(sales_db_path), f"DB not found at: {sales_db_path}"

# creates an instance of FastMCP, a framework used for building Multi-modal Conversational Programs (MCPs) that can use 
# tools (functions) to reason and take actions.
mcp = FastMCP("Sales DB and Github MCP")

# decorator that registers a Python function as a tool for the MCP to call during conversations.
# Enables reasoning → tool calling → response generation
@mcp.tool()
def list_tables() -> list[str]:
    """List all tables in the DB."""
    conn = sqlite3.connect(sales_db_path)
    result = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    conn.close()
    return result

@mcp.tool()
def describe_table(table: str) -> list[dict]:
    """Get schema info for a table."""
    conn = sqlite3.connect(sales_db_path)
    result = [dict(cid=r[0], name=r[1], type=r[2]) for r in conn.execute(f"PRAGMA table_info({table})")]
    conn.close()
    return result

@mcp.tool()
def read_query(query: str) -> list[tuple]:
    """Execute SELECT queries only."""
    if not query.strip().lower().startswith("select"):
        return [("Only SELECT allowed",)]
    conn = sqlite3.connect(sales_db_path)
    result = conn.execute(query).fetchall()
    conn.close()
    return result

@mcp.tool()
def create_repo(repo_name: str) -> str:
    """Creates a GitHub repo under authenticated user's account."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"}
    data = {"name": repo_name, "auto_init": True, "private": False}

    response = requests.post("https://api.github.com/user/repos", json=data, headers=headers)
    if response.status_code == 201:
        return f'The GitHub repository "{repo_name}" has been created successfully.'
    else:
        return f"Failed to create repo: {response.status_code} {response.json()}"


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=6274,
        path="/mcp"
    )
