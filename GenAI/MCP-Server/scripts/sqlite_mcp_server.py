from fastmcp import FastMCP
import sqlite3, os

# Compute DB path
script_dir = os.path.dirname(os.path.abspath(__file__))
sales_db_path = os.path.normpath(os.path.join(script_dir, '..', 'database', 'sales.db'))
assert os.path.exists(sales_db_path), f"DB not found at: {sales_db_path}"

mcp = FastMCP("Sales DB MCP")

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

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=6274,
        path="/mcp"
    )
